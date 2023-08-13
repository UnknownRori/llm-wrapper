mod inference_pipeline;
mod inference_session;

pub use inference_pipeline::{InferencePipeline, InferencePipelineBuilder};
pub use inference_session::{InferencePipelineSession, InferencePipelineSessionBuilder};
pub use llm::InferenceSession;

use llm::{LoadError, LoadProgress};
use thiserror::Error;

pub type ProgressCallback = Box<dyn FnMut(LoadProgress)>;

#[derive(Error, Debug)]
pub enum PipelineBuilderError {
    #[error("Expected to be valid Path to the model : {found:?}")]
    InvalidPathToModel { found: String },

    #[error("Failed to load the model : {0}")]
    FailedToLoadModel(LoadError),
}

#[derive(Error, Debug)]
pub enum SessionInferenceBuilderError {
    #[error("Model is expected to be passed")]
    NoValidModel,
}
