use llm::{models::Gpt2, InferenceRequest, InferenceSessionConfig};
use llm_wrapper::pipeline::InferencePipelineBuilder;

fn main() -> Result<(), &'static str> {
    let mut pipeline = InferencePipelineBuilder::<Gpt2>::new()
        .path(std::path::Path::new("models/gpt-2-774M/ggml-model-f16.bin").into())
        .build()
        .unwrap();

    let mut session = pipeline
        .start_session_builder()
        .session_config(InferenceSessionConfig {
            n_batch: 2,
            n_threads: 1,
            ..Default::default()
        })
        .build()
        .unwrap();

    let a = session.infer(InferenceRequest {
        prompt: "Rust is a cool programming language because".into(),
        parameters: &llm::InferenceParameters::default(),
        play_back_previous_tokens: false,
        maximum_token_count: None,
    });

    dbg!(a);

    Ok(())
}
