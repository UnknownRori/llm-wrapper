use std::io::Write;

use llm::{
    InferenceError, InferenceFeedback, InferenceRequest, InferenceResponse, InferenceSession,
    InferenceSessionConfig, InferenceStats, KnownModel, OutputRequest,
};

use super::SessionInferenceBuilderError;

pub type ResponseCallbackResult = Result<InferenceFeedback, InferenceError>;
pub type ResponseCallback = Box<dyn FnMut(InferenceResponse) -> ResponseCallbackResult>;

pub struct InferencePipelineSession<'a, M: KnownModel> {
    model: &'a M,
    session: InferenceSession,
    output_request: OutputRequest,
    callback: ResponseCallback,
}

impl<'a, M: KnownModel> InferencePipelineSession<'a, M> {
    pub fn new(
        model: &'a M,
        output_request: OutputRequest,
        session_config: InferenceSessionConfig,
        callback: ResponseCallback,
    ) -> Self {
        let session = model.start_session(session_config);

        Self {
            model,
            session,
            output_request,
            callback,
        }
    }

    pub fn infer(
        &mut self,
        request_config: InferenceRequest,
    ) -> Result<InferenceStats, InferenceError> {
        let callback = self.callback.as_mut();

        self.session.infer(
            self.model,
            &mut rand::thread_rng(),
            &request_config,
            &mut self.output_request,
            callback,
        )
    }
}

#[derive(Default)]
pub struct InferencePipelineSessionBuilder<'a, M>
where
    M: KnownModel,
{
    model: Option<&'a M>,
    session_config: InferenceSessionConfig,
    output_request: OutputRequest,
    callback: Option<ResponseCallback>,
}

impl<'a, M: KnownModel> InferencePipelineSessionBuilder<'a, M> {
    pub fn new(model: &'a M) -> Self {
        Self {
            model: Some(model),
            output_request: Default::default(),
            session_config: Default::default(),
            callback: None,
        }
    }

    pub fn model(mut self, model: &'a M) -> Self {
        self.model = Some(model);
        self
    }

    pub fn session_config(mut self, config: InferenceSessionConfig) -> Self {
        self.session_config = config;
        self
    }

    pub fn output_request(mut self, output: OutputRequest) -> Self {
        self.output_request = output;
        self
    }

    pub fn callback(
        mut self,
        callback: Box<dyn FnMut(InferenceResponse) -> ResponseCallbackResult>,
    ) -> Self {
        self.callback = Some(callback);
        self
    }

    pub fn build(self) -> Result<InferencePipelineSession<'a, M>, SessionInferenceBuilderError> {
        let callback = self
            .callback
            .unwrap_or(Box::new(|r| -> ResponseCallbackResult {
                match r {
                    llm::InferenceResponse::PromptToken(t)
                    | llm::InferenceResponse::InferredToken(t) => {
                        print!("{t}");
                        std::io::stdout().flush().unwrap();

                        Ok(llm::InferenceFeedback::Continue)
                    }
                    _ => Ok(llm::InferenceFeedback::Continue),
                }
            }));

        match self.model {
            Some(model) => Ok(InferencePipelineSession::new(
                model,
                self.output_request,
                self.session_config,
                callback,
            )),
            None => Err(SessionInferenceBuilderError::NoValidModel),
        }
    }
}
