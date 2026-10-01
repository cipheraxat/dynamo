// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_backend_common::engine::RoutingHints;
use dynamo_backend_common::{
    BackendError, BootstrapInfo, DisaggregationMode, EngineConfig, FinishReason, GenerateContext,
    GuidedDecodingOptions, LLMEngine, PrefillResult, PreprocessedRequest, testing::mock_context,
};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_runtime::discovery::DiscoverySpec;
use dynamo_sglang_sidecar::SglangSidecarEngine;
use dynamo_sidecar_testkit::{bounded, fixtures};
use dynamo_vllm_sidecar::VllmSidecarEngine;
use futures::StreamExt;

#[allow(dead_code)]
#[path = "support/process.rs"]
mod process;
#[allow(dead_code)]
mod support;

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("native test requires {name}"))
}

#[derive(Clone, Copy)]
enum Backend {
    Vllm,
    Sglang,
}

async fn engine(backend: Backend, endpoint: &str, mode: DisaggregationMode) -> Box<dyn LLMEngine> {
    start_engine(backend, endpoint, mode).await.0
}

async fn start_engine(
    backend: Backend,
    endpoint: &str,
    mode: DisaggregationMode,
) -> (Box<dyn LLMEngine>, EngineConfig) {
    let mut argv = vec![
        "native-test".to_owned(),
        "--grpc-endpoint".to_owned(),
        required(endpoint),
        "--grpc-connections".to_owned(),
        "1".to_owned(),
        "--grpc-startup-deadline-secs".to_owned(),
        "10".to_owned(),
    ];
    if matches!(backend, Backend::Vllm) {
        argv.extend(["--disaggregation-mode".to_owned(), mode.to_string()]);
    }
    let engine = tokio::task::spawn_blocking(move || -> Result<Box<dyn LLMEngine>, _> {
        match backend {
            Backend::Vllm => VllmSidecarEngine::from_args(Some(argv))
                .map(|(engine, _)| Box::new(engine) as Box<dyn LLMEngine>),
            Backend::Sglang => SglangSidecarEngine::from_args(Some(argv))
                .map(|(engine, _)| Box::new(engine) as Box<dyn LLMEngine>),
        }
    })
    .await
    .unwrap()
    .unwrap();
    let config = bounded("native sidecar startup", engine.start(0))
        .await
        .unwrap();
    (engine, config)
}

fn request(max_tokens: u32) -> PreprocessedRequest {
    let mut request =
        fixtures::request(&required("SIDECAR_NATIVE_MODEL"), vec![11; 128], max_tokens);
    request.stop_conditions.ignore_eos = Some(true);
    request.sampling_options.temperature = Some(0.0);
    request
}

async fn metrics_at(endpoint: &str) -> String {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap()
        .get(required(endpoint))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .text()
        .await
        .unwrap()
}

fn metric(body: &str, name: &str) -> f64 {
    metric_if_present(body, name).unwrap_or_else(|| panic!("missing native metric {name}: {body}"))
}

fn metric_if_present(body: &str, name: &str) -> Option<f64> {
    let values: Vec<f64> = body
        .lines()
        .filter(|line| {
            line.strip_prefix(name)
                .is_some_and(|tail| tail.starts_with('{') || tail.starts_with(' '))
        })
        .map(|line| line.rsplit_once(' ').unwrap().1.parse().unwrap())
        .collect();
    (!values.is_empty()).then(|| values.iter().sum())
}

async fn scheduler(backend: Backend, active: bool) {
    scheduler_at(backend, active, "SIDECAR_NATIVE_METRICS").await;
}

async fn scheduler_at(backend: Backend, active: bool, endpoint: &str) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let body = metrics_at(endpoint).await;
        let (running, waiting) = match backend {
            Backend::Vllm => ("vllm:num_requests_running", "vllm:num_requests_waiting"),
            Backend::Sglang => ("sglang:num_running_reqs", "sglang:num_queue_reqs"),
        };
        let running = metric(&body, running);
        let waiting = metric(&body, waiting);
        if (active && running > 0.0) || (!active && running == 0.0 && waiting == 0.0) {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "scheduler active={active}: {body}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn recovery(engine: &dyn LLMEngine) {
    let outputs = fixtures::collect(
        engine,
        request(4),
        GenerateContext::new(mock_context(), None),
    )
    .await;
    let values: Vec<_> = outputs
        .iter()
        .flat_map(|o| o.as_ref().unwrap().token_ids.iter().copied())
        .collect();
    assert_eq!(values.len(), 4);
    dynamo_sidecar_testkit::assert::terminal(outputs, &values, 128, FinishReason::Length);
}

async fn native_logprobs_and_structured_output_are_compatible(backend: Backend) {
    tokio::time::timeout(Duration::from_secs(90), async {
        let engine = engine(
            backend,
            "SIDECAR_NATIVE_GRPC",
            DisaggregationMode::Aggregated,
        )
        .await;
        let mut logprob_request = request(8);
        logprob_request.output_options.logprobs = Some(2);
        logprob_request.output_options.prompt_logprobs = Some(2);
        let outputs = fixtures::collect(
            engine.as_ref(),
            logprob_request,
            GenerateContext::new(mock_context(), None),
        )
        .await;
        let chunks: Vec<_> = outputs.iter().map(|item| item.as_ref().unwrap()).collect();
        assert!(chunks.iter().filter(|o| !o.token_ids.is_empty()).count() > 1);
        let values: Vec<_> = chunks
            .iter()
            .flat_map(|o| o.token_ids.iter().copied())
            .collect();
        assert_eq!(values.len(), 8);
        for chunk in &chunks {
            let selected = chunk.log_probs.as_ref().expect("requested output logprobs");
            let candidates = chunk.top_logprobs.as_ref().expect("requested top logprobs");
            assert_eq!(selected.len(), chunk.token_ids.len());
            assert_eq!(candidates.len(), chunk.token_ids.len());
            for ((token, logprob), row) in chunk.token_ids.iter().zip(selected).zip(candidates) {
                if matches!(backend, Backend::Vllm) {
                    assert_eq!(row.len(), 3, "selected token plus two native candidates");
                    assert_eq!(row[0].token_id, *token);
                    assert_eq!(row[0].logprob, *logprob);
                    assert_eq!(row[0].rank, 1);
                    assert_eq!(row[1].token_id, *token);
                    assert_eq!(row[1].logprob, *logprob);
                    assert_eq!(row[1].rank, 1);
                    assert_eq!(row[2].rank, 2);
                    assert_ne!(row[1].token_id, row[2].token_id);
                } else {
                    assert_eq!(row.len(), 2, "two native candidates");
                    if let Some(selected) = row.iter().find(|entry| entry.token_id == *token) {
                        assert_eq!(selected.logprob, *logprob);
                    }
                    assert_eq!(row[0].logprob, *logprob, "greedy selection may tie");
                    assert_eq!(row[0].rank, 1);
                    assert_eq!(row[1].rank, 2);
                    assert_ne!(row[0].token_id, row[1].token_id);
                }
                for entry in row {
                    assert!(entry.rank > 0);
                    assert!(entry.logprob.is_finite());
                    assert!(entry.logprob > -9999.0 && entry.logprob <= 0.0);
                }
            }
            if chunk.finish_reason.is_none() {
                assert!(chunk.engine_data.is_none());
            }
        }
        let prompt = chunks.last().unwrap().engine_data.as_ref().unwrap()["prompt_logprobs"]
            .as_array()
            .expect("requested terminal prompt logprobs");
        assert_eq!(prompt.len(), 128);
        assert!(prompt[0].is_null());
        for position in &prompt[1..] {
            let entries = position.as_object().unwrap();
            assert!((2..=3).contains(&entries.len()));
            assert!(entries.contains_key("11"), "prompt token association");
            if matches!(backend, Backend::Vllm) {
                for rank in [1, 2] {
                    assert!(
                        entries
                            .values()
                            .any(|entry| entry["rank"].as_u64() == Some(rank))
                    );
                }
            }
            for (token, entry) in entries {
                token.parse::<u32>().expect("candidate token ID");
                if matches!(backend, Backend::Vllm) {
                    assert!(entry["rank"].as_u64().is_some_and(|rank| rank > 0));
                }
                let logprob = entry["logprob"].as_f64().unwrap();
                assert!(logprob.is_finite() && logprob > -9999.0 && logprob <= 0.0);
            }
        }
        dynamo_sidecar_testkit::assert::terminal(outputs, &values, 128, FinishReason::Length);

        let mut structured_request = request(64);
        structured_request.token_ids =
            serde_json::from_str(&required("SIDECAR_NATIVE_STRUCTURED_PROMPT")).unwrap();
        let prompt_tokens = u32::try_from(structured_request.token_ids.len()).unwrap();
        assert!(prompt_tokens > 0);
        structured_request.stop_conditions.ignore_eos = Some(false);
        structured_request.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
            json: Some(serde_json::json!({
                "type": "object",
                "properties": {"ok": {"type": "boolean", "const": true}},
                "required": ["ok"],
                "additionalProperties": false
            })),
            ..Default::default()
        });
        let outputs = tokio::time::timeout(Duration::from_secs(60), async {
            engine
                .generate(
                    structured_request,
                    GenerateContext::new(mock_context(), None),
                )
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        })
        .await
        .expect("native structured-output compilation and generation");
        let mut text = String::new();
        let mut values = Vec::new();
        for output in &outputs {
            let output = output.as_ref().unwrap();
            if matches!(backend, Backend::Vllm) {
                text.push_str(output.text.as_deref().expect("native decoded text"));
            }
            values.extend_from_slice(&output.token_ids);
        }
        assert!(!values.is_empty());
        if matches!(backend, Backend::Vllm) {
            let value: serde_json::Value = serde_json::from_str(&text)
                .unwrap_or_else(|error| panic!("native structured output {text:?}: {error}"));
            assert_eq!(value, serde_json::json!({"ok": true}));
        } else {
            std::fs::write(
                required("SIDECAR_NATIVE_STRUCTURED_TOKENS"),
                serde_json::to_vec(&values).unwrap(),
            )
            .unwrap();
        }
        dynamo_sidecar_testkit::assert::terminal(
            outputs,
            &values,
            prompt_tokens,
            FinishReason::Stop,
        );
        scheduler(backend, false).await;
        engine.cleanup().await.unwrap();
    })
    .await
    .expect("native compatibility scenario timed out");
}

async fn cancellation_and_consumer_drop_release_native_work(backend: Backend) {
    tokio::time::timeout(Duration::from_secs(90), async {
        let engine = engine(
            backend,
            "SIDECAR_NATIVE_GRPC",
            DisaggregationMode::Aggregated,
        )
        .await;
        recovery(engine.as_ref()).await;
        for explicit in [true, false] {
            scheduler(backend, false).await;
            let context = mock_context();
            let mut stream = engine
                .generate(request(4096), GenerateContext::new(context.clone(), None))
                .await
                .unwrap();
            let first = bounded("first native output", stream.next())
                .await
                .unwrap()
                .unwrap();
            assert!(!first.token_ids.is_empty());
            assert_eq!(first.finish_reason, None);
            scheduler(backend, true).await;
            if explicit {
                context.stop_generating();
                let terminal = bounded("native cancellation", stream.next())
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
                assert!(
                    bounded("native cancellation EOF", stream.next())
                        .await
                        .is_none()
                );
            }
            drop(stream);
            scheduler(backend, false).await;
            recovery(engine.as_ref()).await;
        }
        engine.cleanup().await.unwrap();
    })
    .await
    .expect("native cancellation scenario timed out");
}

fn vllm_transferred_bytes() -> u64 {
    std::fs::read_to_string(required("SIDECAR_NATIVE_TRANSFER_PROBE"))
        .expect("native NIXL telemetry must be recorded")
        .lines()
        .map(|line| {
            serde_json::from_str::<serde_json::Value>(line).unwrap()["bytes"]
                .as_u64()
                .unwrap()
        })
        .sum()
}

#[tokio::test]
async fn vllm_handoff_transfers_native_kv() {
    tokio::time::timeout(Duration::from_secs(90), async {
        let prefill = engine(
            Backend::Vllm,
            "SIDECAR_NATIVE_PREFILL_GRPC",
            DisaggregationMode::Prefill,
        )
        .await;
        let decode = engine(
            Backend::Vllm,
            "SIDECAR_NATIVE_GRPC",
            DisaggregationMode::Decode,
        )
        .await;
        for (token, is_cancelled) in [(11, true), (12, false)] {
            let mut prefill_request = request(8);
            prefill_request.token_ids = vec![token; 128].into();
            let mut decode_request = prefill_request.clone();
            let outputs = fixtures::collect(
                prefill.as_ref(),
                prefill_request,
                GenerateContext::new(mock_context(), None),
            )
            .await;
            let outputs: Vec<_> = outputs.into_iter().collect::<Result<_, _>>().unwrap();
            assert!(outputs.iter().all(|o| o.token_ids.is_empty()));
            let last = outputs.last().unwrap();
            assert_eq!(last.finish_reason, Some(FinishReason::Length));
            let handoff = last
                .disaggregated_params
                .clone()
                .expect("native prefill handoff");
            assert!(
                handoff["remote_engine_id"]
                    .as_str()
                    .is_some_and(|id| !id.is_empty())
            );
            assert!(
                handoff["remote_block_ids"]
                    .as_array()
                    .is_some_and(|ids| !ids.is_empty())
            );
            decode_request.prefill_result = Some(PrefillResult {
                disaggregated_params: handoff,
                prompt_tokens_details: None,
            });
            let before = vllm_transferred_bytes();
            let context = mock_context();
            if is_cancelled {
                // Decode must finish the transfer even when cancelled before submission.
                context.stop_generating();
            }
            let outputs = fixtures::collect(
                decode.as_ref(),
                decode_request,
                GenerateContext::new(context, None),
            )
            .await;
            let values: Vec<_> = outputs
                .iter()
                .flat_map(|o| o.as_ref().unwrap().token_ids.iter().copied())
                .collect();
            if is_cancelled {
                assert!(values.is_empty());
                let outputs: Vec<_> = outputs.into_iter().collect::<Result<_, _>>().unwrap();
                assert_eq!(outputs.len(), 1);
                assert_eq!(outputs[0].finish_reason, Some(FinishReason::Cancelled));
            } else {
                assert_eq!(values.len(), 8);
                dynamo_sidecar_testkit::assert::terminal(
                    outputs,
                    &values,
                    128,
                    FinishReason::Length,
                );
            }
            bounded("native completed KV transfer", async {
                while vllm_transferred_bytes() <= before {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await;
            scheduler(Backend::Vllm, false).await;
            scheduler_at(Backend::Vllm, false, "SIDECAR_NATIVE_PREFILL_METRICS").await;
        }
        prefill.cleanup().await.unwrap();
        decode.cleanup().await.unwrap();
    })
    .await
    .expect("native handoff scenario timed out");
}

#[tokio::test]
async fn vllm_native_logprobs_and_structured_output_are_compatible() {
    native_logprobs_and_structured_output_are_compatible(Backend::Vllm).await;
}

#[tokio::test]
async fn sglang_native_logprobs_and_structured_output_are_compatible() {
    native_logprobs_and_structured_output_are_compatible(Backend::Sglang).await;
}

#[tokio::test]
async fn vllm_cancellation_and_consumer_drop_release_native_work() {
    cancellation_and_consumer_drop_release_native_work(Backend::Vllm).await;
}

#[tokio::test]
async fn sglang_cancellation_and_consumer_drop_release_native_work() {
    cancellation_and_consumer_drop_release_native_work(Backend::Sglang).await;
}

async fn sglang_transfer_queue(has_work: bool) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let url = format!(
        "{}/v1/loads?include=core",
        required("SIDECAR_NATIVE_METRICS").trim_end_matches("/metrics")
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let body: serde_json::Value = client
            .get(&url)
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap();
        let loads = body["loads"]
            .as_array()
            .expect("native scheduler load snapshots");
        assert!(!loads.is_empty());
        let sum = |name: &str| {
            loads
                .iter()
                .map(|load| load[name].as_u64().unwrap())
                .sum::<u64>()
        };
        let running = sum("num_running_reqs");
        let waiting = sum("num_waiting_reqs");
        let is_awaiting_kv = sum("num_total_tokens") > sum("num_active_tokens");
        if running == 0
            && if has_work {
                waiting > 0 && is_awaiting_kv
            } else {
                waiting == 0
            }
        {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "native transfer queue has_work={has_work}: {body}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn sglang_cancel_during_transfer_wait(decode: &dyn LLMEngine, bootstrap: BootstrapInfo) {
    sglang_transfer_queue(false).await;
    let context = mock_context();
    let mut pending = request(8);
    pending.bootstrap_info = Some(bootstrap);
    let mut stream = decode
        .generate(pending, GenerateContext::new(context.clone(), None))
        .await
        .unwrap();
    tokio::select! {
        _ = sglang_transfer_queue(true) => {},
        output = stream.next() => panic!("decode produced output without its prefill peer: {output:?}"),
    }
    context.stop_generating();
    let outputs = bounded(
        "native transfer-wait cancellation",
        stream.collect::<Vec<_>>(),
    )
    .await;
    dynamo_sidecar_testkit::assert::terminal(outputs, &[], 128, FinishReason::Cancelled);
    sglang_transfer_queue(false).await;
    scheduler(Backend::Sglang, false).await;
}

#[tokio::test]
async fn sglang_handoff_transfers_native_kv() {
    tokio::time::timeout(Duration::from_secs(90), async {
        let (prefill, config) = start_engine(
            Backend::Sglang,
            "SIDECAR_NATIVE_PREFILL_GRPC",
            DisaggregationMode::Prefill,
        )
        .await;
        let decode = engine(
            Backend::Sglang,
            "SIDECAR_NATIVE_GRPC",
            DisaggregationMode::Decode,
        )
        .await;
        let registration = config.llm.expect("native prefill registration");
        let host = registration.bootstrap_host.expect("native bootstrap host");
        let port = registration.bootstrap_port.expect("native bootstrap port");
        assert!(!host.is_empty());
        assert!(port > 0);
        for room in [17, 18] {
            let bootstrap = BootstrapInfo {
                bootstrap_host: host.clone(),
                bootstrap_port: port,
                bootstrap_room: room,
                handoff_id: None,
            };
            let mut prefill_request = request(8);
            prefill_request.token_ids = vec![11 + room as u32; 128].into();
            prefill_request.bootstrap_info = Some(bootstrap.clone());
            let mut decode_request = request(8);
            decode_request
                .token_ids
                .clone_from(&prefill_request.token_ids);
            decode_request.bootstrap_info = Some(bootstrap);
            let before = metric_if_present(
                &metrics_at("SIDECAR_NATIVE_PREFILL_METRICS").await,
                "sglang:kv_transfer_total_mb_sum",
            )
            .unwrap_or_default();
            let (prefill_outputs, decode_outputs) = tokio::join!(
                fixtures::collect(
                    prefill.as_ref(),
                    prefill_request,
                    GenerateContext::new(mock_context(), None)
                ),
                fixtures::collect(
                    decode.as_ref(),
                    decode_request,
                    GenerateContext::new(mock_context(), None)
                ),
            );
            let handoffs: Vec<_> = prefill_outputs
                .iter()
                .filter_map(|output| output.as_ref().unwrap().disaggregated_params.as_ref())
                .collect();
            assert_eq!(handoffs.len(), 1);
            assert_eq!(
                handoffs[0],
                &serde_json::json!({
                    "bootstrap_host": host, "bootstrap_port": port, "bootstrap_room": room,
                })
            );
            dynamo_sidecar_testkit::assert::terminal(
                prefill_outputs,
                &[],
                128,
                FinishReason::Length,
            );
            let values: Vec<_> = decode_outputs
                .iter()
                .flat_map(|output| output.as_ref().unwrap().token_ids.iter().copied())
                .collect();
            assert_eq!(values.len(), 8);
            dynamo_sidecar_testkit::assert::terminal(
                decode_outputs,
                &values,
                128,
                FinishReason::Length,
            );
            bounded("native KV transfer telemetry", async {
                loop {
                    if metric_if_present(
                        &metrics_at("SIDECAR_NATIVE_PREFILL_METRICS").await,
                        "sglang:kv_transfer_total_mb_sum",
                    )
                    .unwrap_or_default()
                        > before
                    {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await;
            scheduler(Backend::Sglang, false).await;
            scheduler_at(Backend::Sglang, false, "SIDECAR_NATIVE_PREFILL_METRICS").await;
            if room == 17 {
                sglang_cancel_during_transfer_wait(
                    decode.as_ref(),
                    BootstrapInfo {
                        bootstrap_host: host.clone(),
                        bootstrap_port: port,
                        bootstrap_room: 19,
                        handoff_id: None,
                    },
                )
                .await;
            }
        }
        prefill.cleanup().await.unwrap();
        decode.cleanup().await.unwrap();
    })
    .await
    .expect("native SGLang bootstrap handoff timed out");
}

fn lora_request(adapter: Option<&str>) -> PreprocessedRequest {
    let mut request = request(4);
    request.token_ids = serde_json::from_str(&required("SIDECAR_NATIVE_LORA_PROMPT")).unwrap();
    request.output_options.logprobs = Some(1);
    request.routing = Some(RoutingHints {
        lora_name: adapter.map(str::to_owned),
        ..Default::default()
    });
    request
}

async fn lora_sample(engine: &dyn LLMEngine, adapter: Option<&str>) -> Vec<(u32, f64)> {
    let request = lora_request(adapter);
    let prompt_tokens = request.token_ids.len() as u32;
    let outputs =
        fixtures::collect(engine, request, GenerateContext::new(mock_context(), None)).await;
    let sample: Vec<_> = outputs
        .iter()
        .flat_map(|output| {
            let output = output.as_ref().unwrap();
            output
                .token_ids
                .iter()
                .copied()
                .zip(output.log_probs.as_ref().unwrap().iter().copied())
        })
        .collect();
    assert_eq!(sample.len(), 4);
    assert!(sample.iter().all(|(_, logprob)| logprob.is_finite()));
    let tokens: Vec<_> = sample.iter().map(|(token, _)| *token).collect();
    dynamo_sidecar_testkit::assert::terminal(outputs, &tokens, prompt_tokens, FinishReason::Length);
    sample
}

#[tokio::test]
async fn vllm_lora_lifecycle_selects_native_adapter() {
    tokio::time::timeout(Duration::from_secs(90), async {
        let env = process::Environment::new().await;
        let engine = engine(
            Backend::Vllm,
            "SIDECAR_NATIVE_GRPC",
            DisaggregationMode::Aggregated,
        )
        .await;
        let endpoint = env.endpoint("backend");
        let base = ModelDeploymentCard::with_name_only(&required("SIDECAR_NATIVE_MODEL"));
        env.runtime
            .discovery()
            .register(
                DiscoverySpec::from_model(
                    env.namespace.clone(),
                    "backend".into(),
                    "generate".into(),
                    &base,
                )
                .unwrap(),
            )
            .await
            .unwrap();
        engine.on_endpoint_ready(endpoint).await.unwrap();
        let baseline = lora_sample(engine.as_ref(), None).await;
        let name = "native-adapter";
        let loaded = engine
            .engine_update(
                "load_lora".into(),
                serde_json::json!({
                    "lora_name": name,
                    "source": {"uri": format!("file://{}", required("SIDECAR_NATIVE_LORA_PATH"))}
                }),
            )
            .await
            .unwrap();
        assert_eq!(loaded["status"], "success", "{loaded}");
        let id = loaded["lora_id"].as_u64().expect("native adapter ID");
        assert!(id > 0);
        let inventory = engine
            .engine_update("list_loras".into(), serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(inventory["loras"], serde_json::json!({name: id}));
        let cards = env.cards().await;
        assert_eq!(cards.len(), 2);
        assert!(cards.iter().any(|card| {
            card.user_data
                .as_ref()
                .is_some_and(|data| data["lora_id"] == id)
        }));
        let adapted = lora_sample(engine.as_ref(), Some(name)).await;
        assert_ne!(
            adapted, baseline,
            "the trained adapter must change native token scores"
        );
        let unloaded = engine
            .engine_update("unload_lora".into(), serde_json::json!({"lora_name": name}))
            .await
            .unwrap();
        assert_eq!(unloaded["status"], "success", "{unloaded}");
        let inventory = engine
            .engine_update("list_loras".into(), serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(inventory["loras"], serde_json::json!({}));
        assert_eq!(env.cards().await.len(), 1);
        let rejected = fixtures::collect(
            engine.as_ref(),
            lora_request(Some(name)),
            GenerateContext::new(mock_context(), None),
        )
        .await;
        let error =
            dynamo_sidecar_testkit::assert::failure(rejected, &[], BackendError::InvalidArgument);
        assert!(error.to_string().contains("unknown model or LoRA adapter"));
        lora_sample(engine.as_ref(), None).await;
        scheduler(Backend::Vllm, false).await;
        engine.cleanup().await.unwrap();
    })
    .await
    .expect("native LoRA lifecycle timed out");
}

#[tokio::test]
async fn sglang_preloaded_lora_selects_native_adapter() {
    tokio::time::timeout(Duration::from_secs(90), async {
        let engine = engine(
            Backend::Sglang,
            "SIDECAR_NATIVE_GRPC",
            DisaggregationMode::Aggregated,
        )
        .await;
        let baseline = lora_sample(engine.as_ref(), None).await;
        let adapted = lora_sample(engine.as_ref(), Some("native-adapter")).await;
        assert_ne!(
            adapted, baseline,
            "the preloaded adapter must change native token scores"
        );
        let rejected = fixtures::collect(
            engine.as_ref(),
            lora_request(Some("missing-adapter")),
            GenerateContext::new(mock_context(), None),
        )
        .await;
        assert!(
            rejected.iter().any(Result::is_err),
            "unknown native adapter must fail"
        );
        lora_sample(engine.as_ref(), None).await;
        scheduler(Backend::Sglang, false).await;
        engine.cleanup().await.unwrap();
    })
    .await
    .expect("native preloaded LoRA timed out");
}

fn sglang_http_request(max_tokens: u32) -> PreprocessedRequest {
    let mut request = request(max_tokens);
    request.extra_args = Some(serde_json::json!({"sglang_tito": {
        "sampling_params": {"max_new_tokens": max_tokens, "ignore_eos": true, "temperature": 0.0},
        "return_logprob": true,
        "top_logprobs_num": 2
    }}));
    request
}

async fn sglang_http_recovery(engine: &dyn LLMEngine) {
    let outputs = fixtures::collect(
        engine,
        sglang_http_request(4),
        GenerateContext::new(mock_context(), None),
    )
    .await;
    let outputs: Vec<_> = outputs.into_iter().collect::<Result<_, _>>().unwrap();
    assert!(
        outputs.len() > 1,
        "native HTTP must stream incremental output"
    );
    let mut token_count = 0;
    for (index, output) in outputs.iter().enumerate() {
        assert!(
            output.token_ids.is_empty(),
            "native HTTP keeps its payload opaque"
        );
        let raw = &output.engine_data.as_ref().unwrap()["sglang_response"];
        token_count += raw["output_ids"].as_array().unwrap().len();
        assert!(raw["meta_info"]["output_token_logprobs"].is_array());
        if index + 1 < outputs.len() {
            assert!(output.finish_reason.is_none());
            assert!(raw["meta_info"]["finish_reason"].is_null());
        } else {
            assert_eq!(output.finish_reason, Some(FinishReason::Stop));
            assert_eq!(raw["meta_info"]["finish_reason"]["type"], "length");
            assert_eq!(raw["meta_info"]["prompt_tokens"], 128);
            assert_eq!(raw["meta_info"]["completion_tokens"], 4);
        }
    }
    assert_eq!(token_count, 4);
}

#[tokio::test]
async fn sglang_native_http_stream_cancel_and_drop_recover() {
    tokio::time::timeout(Duration::from_secs(90), async {
        let engine = engine(
            Backend::Sglang,
            "SIDECAR_NATIVE_GRPC",
            DisaggregationMode::Aggregated,
        )
        .await;
        sglang_http_recovery(engine.as_ref()).await;
        for is_explicit in [true, false] {
            scheduler(Backend::Sglang, false).await;
            let context = mock_context();
            let mut stream = engine
                .generate(
                    sglang_http_request(4096),
                    GenerateContext::new(context.clone(), None),
                )
                .await
                .unwrap();
            let first = bounded("first native HTTP output", stream.next())
                .await
                .unwrap()
                .unwrap();
            assert!(first.finish_reason.is_none());
            assert!(
                !first.engine_data.as_ref().unwrap()["sglang_response"]["output_ids"]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
            scheduler(Backend::Sglang, true).await;
            if is_explicit {
                context.stop_generating();
                let error = bounded("native HTTP cancellation", stream.next())
                    .await
                    .unwrap()
                    .unwrap_err();
                assert_eq!(
                    error.error_type(),
                    dynamo_backend_common::ErrorType::Backend(BackendError::Cancelled)
                );
                assert!(
                    bounded("native HTTP cancellation EOF", stream.next())
                        .await
                        .is_none()
                );
            }
            drop(stream);
            scheduler(Backend::Sglang, false).await;
            sglang_http_recovery(engine.as_ref()).await;
        }
        engine.cleanup().await.unwrap();
    })
    .await
    .expect("native HTTP cancellation timed out");
}
