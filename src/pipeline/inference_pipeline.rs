use std::path::PathBuf;

use llm::{
    load as load_model, load_progress_callback_stdout, KnownModel, LoadProgress, ModelParameters,
    TokenizerSource,
};

use super::{InferencePipelineSessionBuilder, PipelineBuilderError, ProgressCallback};

pub struct InferencePipeline<M: KnownModel> {
    model: M,
}

impl<M: KnownModel> InferencePipeline<M> {
    pub fn new(model: M) -> Self {
        Self { model }
    }

    /// Get the model
    pub fn model(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn start_session_builder(&mut self) -> InferencePipelineSessionBuilder<'_, M> {
        InferencePipelineSessionBuilder::new(&(self.model))
    }
}

pub struct InferencePipelineBuilder<M: KnownModel> {
    path: Option<PathBuf>,
    tokenizer: Option<TokenizerSource>,
    model_parameter: Option<ModelParameters>,
    load_progress_callback: Option<ProgressCallback>,
    ghost: std::marker::PhantomData<M>,
}

impl<M: KnownModel> Default for InferencePipelineBuilder<M> {
    fn default() -> Self {
        Self {
            path: None,
            tokenizer: None,
            model_parameter: None,
            load_progress_callback: None,
            ghost: std::marker::PhantomData,
        }
    }
}

impl<M: KnownModel> InferencePipelineBuilder<M> {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn path(mut self, path: PathBuf) -> Self {
        self.path = Some(path);
        self
    }

    pub fn tokenizer(mut self, tokenizer: TokenizerSource) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    pub fn load_progress_callback(mut self, callback: Box<dyn FnMut(LoadProgress)>) -> Self {
        self.load_progress_callback = Some(callback);
        self
    }

    pub fn model_parameter(mut self, param: ModelParameters) -> Self {
        self.model_parameter = Some(param);
        self
    }

    pub fn build(self) -> Result<InferencePipeline<M>, PipelineBuilderError> {
        if self.path.is_none() {
            return Err(PipelineBuilderError::InvalidPathToModel {
                found: "[Empty]".to_owned(),
            });
        }

        let tokenizer = match self.tokenizer {
            Some(tokenizer) => tokenizer,
            None => TokenizerSource::Embedded,
        };
        let model_param = match self.model_parameter {
            Some(param) => param,
            None => Default::default(),
        };
        let load_progress_callback = match self.load_progress_callback {
            Some(callback) => callback,
            None => Box::new(load_progress_callback_stdout),
        };

        let model = load_model(
            self.path.unwrap().as_path(),
            tokenizer,
            model_param,
            load_progress_callback,
        )
        .map_err(PipelineBuilderError::FailedToLoadModel)?;

        let model = InferencePipeline::<M>::new(model);

        Ok(model)
    }
}
