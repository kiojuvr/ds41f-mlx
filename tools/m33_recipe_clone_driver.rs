// Release Rust clone timing for the exact native binding's SemanticSession type.
// Source-only diagnostics: no Python ABI timer, MTP target or DSpark execution.
use std::{io::{self, Read}, sync::Arc, time::Instant};
use deepseek_recipe::stream::{SemanticSession, SemanticTerminalKind};
use deepseek_recipe::stream::state_machine::{ParsingOptions, ReasoningStage};
use serde_json::{json, Value};
use tokenizers::Tokenizer;

fn stats(values: &mut [u64]) -> Value {
    values.sort_unstable();
    json!({"samples": values.len(), "median_ns": values[values.len()/2],
        "p95_ns": values[(values.len()*95/100).min(values.len()-1)],
        "max_ns": values[values.len()-1]})
}

fn terminal(p: Option<&deepseek_recipe::stream::SemanticTerminal>) -> Value {
    match p {
        None => Value::Null,
        Some(t) => json!({"kind": match t.kind {
            SemanticTerminalKind::StopSequence => "STOP_SEQUENCE",
            SemanticTerminalKind::DsmlToolCallBlockEnd => "DSML_TOOL_CALL_BLOCK_END",
        }, "source_start": t.source_start, "source_end": t.source_end}),
    }
}

fn main() {
    let tokenizer = Arc::new(Tokenizer::from_file(std::env::args().nth(1).unwrap()).unwrap());
    let mut data = String::new(); io::stdin().read_to_string(&mut data).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&data).unwrap();
    let mut output = Vec::new();
    let mut all_latencies = Vec::new();
    let mut all_clones = Vec::new();
    let repeats = 200;
    for case in cases {
        let options = ParsingOptions {
            parse_json_output: case["json"].as_bool().unwrap_or(false),
            reasoning_initial_stage: if case["reasoning"].as_bool().unwrap_or(false) {
                Some(ReasoningStage::Reasoning)
            } else { None },
            stop_sequences: case["stops"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().into()).collect(),
            ..Default::default()
        };
        let tokens = if let Some(ids) = case["tokens"].as_array() {
            ids.iter().map(|v| v.as_u64().unwrap() as u32).collect::<Vec<_>>()
        } else { tokenizer.encode(case["text"].as_str().unwrap(), false).unwrap().get_ids().to_vec() };
        let mut canonical = SemanticSession::new(options).with_tokenizer(tokenizer.clone());
        let mut rows = Vec::new();
        let mut emitted = 0;
        for window in tokens.chunks(6) {
            if canonical.terminal().is_some() { break; }
            let bytes_before = canonical.source_bytes();
            let pending_before = canonical.pending_ids().to_vec();
            let predicted = canonical.preview_tokens(window).unwrap();
            assert!(predicted.mapping_exact);
            let mut latencies = Vec::new();
            let mut clones = Vec::new();
            for _ in 0..repeats {
                let start = Instant::now();
                let fork = std::hint::black_box(canonical.clone());
                clones.push(start.elapsed().as_nanos() as u64);
                drop(fork);
                let start = Instant::now();
                let result = canonical.preview_tokens(window).unwrap();
                latencies.push(start.elapsed().as_nanos() as u64);
                assert_eq!(result, predicted);
            }
            assert_eq!(canonical.source_bytes(), bytes_before);
            assert_eq!(canonical.pending_ids(), pending_before);
            assert!(canonical.terminal().is_none());
            let mut completing = None;
            for (index, token) in window.iter().enumerate() {
                canonical.token(*token).unwrap();
                emitted += 1;
                if canonical.terminal().is_some() { completing = Some(index); break; }
            }
            assert_eq!(completing, predicted.completing_token_index);
            assert_eq!(canonical.terminal(), predicted.terminal.as_ref());
            all_latencies.extend_from_slice(&latencies);
            all_clones.extend_from_slice(&clones);
            rows.push(json!({"candidate_ids": window, "canonical_emitted_ids_after": emitted,
                "source_bytes_before": bytes_before, "source_bytes_after": canonical.source_bytes(),
                "pending_canonical_ids_before": pending_before,
                "pending_canonical_ids_after": canonical.pending_ids(),
                "safe_token_count": predicted.safe_token_count,
                "completing_token_index": predicted.completing_token_index,
                "mapping_exact": predicted.mapping_exact,
                "terminal": terminal(predicted.terminal.as_ref()),
                "canonical_terminal": terminal(canonical.terminal()),
                "candidate_ids_processed": predicted.candidate_ids_processed,
                "max_pending_ids": predicted.max_pending_ids,
                "preview_latency": stats(&mut latencies), "clone_latency": stats(&mut clones),
                "target_frontier": Value::Null, "dspark_frontier": Value::Null}));
        }
        output.push(json!({"name": case["name"], "rows": rows}));
    }
    println!("{}", serde_json::to_string_pretty(&json!({
        "scope": "release Rust same-core SemanticSession clone (black-boxed) and preview; NOT separate Python ABI clone timing or live MTP",
        "candidate_window_ids": 6, "preview_calls_per_diagnostic_row": repeats+1,
        "repeat_count_for_timing": repeats, "preview_latency": stats(&mut all_latencies),
        "clone_latency": stats(&mut all_clones), "cases": output,
        "model_history_replay": 0, "parser_history_replay": 0,
        "zero_counter_scope": "this driver runs no model, and its parser is incremental"})).unwrap());
}
