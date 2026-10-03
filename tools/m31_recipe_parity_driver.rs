// Standalone diagnostic driver compiled separately against pristine and patched
// recipe crates. Contains no decoder/parser implementation or MTP policy.
use std::{io::{self, Read}, sync::Arc};
use deepseek_recipe::stream::{ChunkGenerator, InferenceChunk, InferenceFinishReason,
    OutputChunk, StreamProcessor};
use deepseek_recipe::stream::state_machine::{ParsingOptions, ReasoningStage};
use deepseek_recipe::openai::chat_completion::response::ChatCompletionChunkGenerator;
use serde_json::{json, Value};
use tokenizers::Tokenizer;
use tokio_stream::StreamExt;

struct Echo;
impl ChunkGenerator for Echo {
    type Chunk = OutputChunk;
    async fn generate(&mut self, value: OutputChunk) -> Vec<OutputChunk> { vec![value] }
}

fn input(case: &Value, mode: &str, tokenizer: &Tokenizer) -> Vec<InferenceChunk> {
    let text = case["text"].as_str().unwrap_or("");
    let mut chunks = match mode {
        "token" => {
            let ids = if let Some(ids) = case["tokens"].as_array() {
                ids.iter().map(|v| v.as_u64().unwrap() as u32).collect::<Vec<_>>()
            } else { tokenizer.encode(text, false).unwrap().get_ids().to_vec() };
            ids.into_iter().map(|token_id| InferenceChunk::Token { token_id }).collect()
        },
        "chars" => text.chars().map(|ch| InferenceChunk::Text {
            content: ch.to_string(), content_tokens: 1 }).collect(),
        "text" => vec![InferenceChunk::Text { content: text.into(), content_tokens: 1 }],
        _ => panic!("unknown input mode"),
    };
    let reason = match case["finish"].as_str().unwrap_or("stop") {
        "stop" => Some(InferenceFinishReason::Stop),
        "length" => Some(InferenceFinishReason::Length),
        "eof" => None,
        _ => panic!("unknown finish"),
    };
    if let Some(finish_reason) = reason { chunks.push(InferenceChunk::Finish { finish_reason }); }
    chunks
}

fn options(case: &Value) -> ParsingOptions {
    ParsingOptions {
        parse_tool_calls: true,
        parse_json_output: case["json"].as_bool().unwrap_or(false),
        reasoning_initial_stage: if case["reasoning"].as_bool().unwrap_or(false) {
            Some(ReasoningStage::Reasoning)
        } else { None },
        stop_sequences: case["stops"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().into()).collect(),
        ..Default::default()
    }
}

#[tokio::main]
async fn main() {
    let tokenizer = Arc::new(Tokenizer::from_file(std::env::args().nth(1).unwrap()).unwrap());
    let mut data = String::new(); io::stdin().read_to_string(&mut data).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&data).unwrap();
    let mut output = Vec::new();
    for case in cases {
        for mode in ["token", "text", "chars"] {
            if case["tokens"].is_array() && mode != "token" { continue; }
            let p = StreamProcessor::new(Echo, options(&case)).with_tokenizer(tokenizer.clone());
            let stream = p.process(tokio_stream::iter(input(&case, mode, &tokenizer)));
            tokio::pin!(stream);
            let mut raw = Vec::new();
            while let Some(event) = stream.next().await { raw.push(format!("{:?}", event.unwrap())); }
            let g = ChatCompletionChunkGenerator::new("m31-parity".into(), "v41".into(), true,
                case["reasoning"].as_bool().unwrap_or(false));
            let p = StreamProcessor::new(g, options(&case)).with_tokenizer(tokenizer.clone());
            let stream = p.process(tokio_stream::iter(input(&case, mode, &tokenizer)));
            tokio::pin!(stream);
            let mut events = Vec::new();
            while let Some(event) = stream.next().await {
                let mut event = serde_json::to_value(event.unwrap()).unwrap();
                // Clock metadata is not parser semantics; hold it fixed.
                event["created"] = json!(0);
                events.push(event);
            }
            output.push(json!({"name": case["name"], "input_mode": mode,
                "raw_events": raw, "chat_events": events}));
        }
    }
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
