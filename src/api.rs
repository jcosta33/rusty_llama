// The `use` statement in Rust is similar to `import` in JavaScript. It's used to bring libraries or modules into scope for use in the current file.
use cfg_if::cfg_if;

/// Uses a conditional compilation block to include code based on whether the "ssr" (server-side rendering) feature is enabled.
/// In Rust, conditional compilation is managed at compile-time, allowing for more optimized builds depending on the features enabled.
cfg_if! {
    // This checks if the "ssr" feature flag is enabled in the Cargo.toml file. If so, the enclosed code is compiled and included.
    if #[cfg(feature = "ssr")] {
        /// Declares global immutable variables accessible throughout the lifetime of the program.
        /// `static` indicates that the variable's memory allocation is static, lasting for the entire run of the program.
        /// In contrast, JavaScript global variables are declared outside of functions but don't use a keyword like `static`.
        static ASSISTANT_NAME: &str = "### Assistant";
        static USER_NAME: &str = "### Human";

        // Importing modules and types from the standard library and external crates, analogous to JavaScript imports but more specific in scope.
        use std::convert::Infallible; // A type for operations that cannot fail.
        use actix_web::web; // Actix web framework's utilities for handling web requests.
        use std::sync::Arc; // A thread-safe way to share ownership of immutable data across threads.
        use llm::models::Llama; // A hypothetical data model from the `llm` crate.
        use llm::KnownModel; // A trait indicating the model's capabilities.
        use actix_web::HttpRequest; // Represents client requests to the server.
        use actix_web::HttpResponse; // For constructing server responses to send back to clients.
        use actix_web::web::Payload; // The payload of a request, such as form data or JSON.
        use actix_web::Error; // General error type for handling issues in request processing.
        use actix_ws::Message as Msg; // WebSocket message handling, with `as` renaming it for clarity.
        use futures::stream::StreamExt; // Extensions for working with asynchronous streams.
        use leptos::*; // Assuming `leptos` is a framework or utility library, importing everything from it.

        /// Performs AI model inference based on a user's message and sends the result over a WebSocket connection.
        ///
        /// # Arguments
        /// * `model` - A shared, thread-safe reference to the AI model.
        /// * `inference_session` - A mutable reference to the inference session, allowing the session to be updated.
        /// * `user_message` - The message from the user, wrapped in a `String`.
        /// * `tx` - The transmitter part of a channel for sending the inference result back to the client.
        ///
        /// # Returns
        /// This function returns a `Result` type, which is either `Ok(())` indicating success, or `Err(ServerFnError)` indicating an error occurred.
        pub fn infer(
            model: Arc<Llama>,
            inference_session: &mut llm::InferenceSession,
            user_message: &String,
            tx: tokio::sync::mpsc::Sender<String>
        ) -> Result<(), ServerFnError> {
            // Bringing the Tokio runtime into scope specifically for this function. Required for running asynchronous tasks synchronously.
            use tokio::runtime::Runtime;

            /// Creates a new Tokio runtime, panicking if this fails. A runtime is required for executing asynchronous code.
            /// `expect` is a method that panics with the specified message if the called function returns an `Err`.
            let mut runtime = Runtime::new().expect("issue creating tokio runtime");

            /// Performs the inference by calling the `infer` method on the session, passing in the model, a random number generator,
            /// and the inference request constructed from the user's message.
            /// `.unwrap_or_else(|e| panic!("{e}"))` will panic if the inference fails, providing the error message.
            inference_session
                .infer(
                    model.as_ref(), // Converts `Arc<Llama>` into a standard reference `&Llama`, allowing the model to be used without taking ownership.
                    &mut rand::thread_rng(), // Creates a thread-local random number generator for the inference.
                    &(llm::InferenceRequest {
                        // Constructs the inference request, using `format!` to interpolate user and character names into the prompt.
                        prompt: format!("{USER_NAME}\n{user_message}\n{ASSISTANT_NAME}:")
                            .as_str() // Converts the formatted string into a string slice (`&str`).
                            .into(), // Converts the string slice into a type compatible with the inference request.
                        parameters: &llm::InferenceParameters::default(), // Uses default parameters for the inference.
                        play_back_previous_tokens: false, // Indicates whether to play back previous tokens during inference.
                        maximum_token_count: None, // No maximum token count is specified, allowing the model to decide.
                    }),
                    &mut Default::default(), // Uses a default empty state for any additional required state.
                    inference_callback(
                        String::from(USER_NAME),
                        &mut String::new(),
                        tx,
                        &mut runtime
                    ) // The callback function to process inference results.
                )
                .unwrap_or_else(|e| panic!("{e}")); // Handles any potential errors from the inference process.

            // If the function reaches this point, it means inference succeeded without errors, returning an `Ok` result.
            Ok(())
        }

        /// Sets up a new inference session with the provided model, returning the session object.
        /// This function demonstrates initializing a session with a specific context or persona for the AI.
        ///
        /// # Arguments
        /// * `model` - A shared, thread-safe reference to the Llama model.
        ///
        /// # Returns
        /// Returns a new `InferenceSession` object, ready for performing inferences.
        fn session_setup(model: Arc<Llama>) -> llm::InferenceSession {
            // Static string representing the initial context for the AI conversation.
            let persona = "A chat between a human and an assistant.";
            // Formatted string representing a hypothetical initial exchange, not used further in this snippet.
            let _history = format!(
                "{CHARACTER_NAME}:Hello - How may I help you today?\n\
                {USER_NAME}:What is the capital of France?\n\
                {CHARACTER_NAME}: is the capital of France.\n"
            );

            // Starts a new session with the AI model, using default settings.
            let mut session = model.start_session(Default::default());
            // Feeds the initial prompt or context into the session, preparing it for further inferences.
            session
                .feed_prompt(
                    model.as_ref(), // As before, converts `Arc<Llama>` to `&Llama`.
                    format!("{persona}").as_str(), // Uses the `persona` variable as the initial prompt.
                    &mut Default::default(), // Uses default settings for any additional state.
                    llm::feed_prompt_callback(|_| {
                        // Defines a callback for handling the AI's response to the initial prompt.
                        Ok::<llm::InferenceFeedback, Infallible>(llm::InferenceFeedback::Continue)
                        // Indicates that the session should continue after receiving the initial response.
                    })
                )
                .expect("Failed to ingest initial prompt."); // Handles any errors that occur during prompt feeding.

            // Returns the session, now ready for performing inferences based on user input.
            session
        }
        /// Creates a callback function for processing inference responses, deciding when to stop based on a stop sequence.
        ///
        /// # Arguments
        /// * `stop_sequence` - A specific string sequence indicating when to stop the inference.
        /// * `buf` - A buffer for accumulating tokens from the inference response.
        /// * `tx` - A transmitter for sending messages over a channel.
        /// * `runtime` - The Tokio runtime for executing asynchronous tasks.
        ///
        /// # Returns
        /// A closure that processes inference responses and decides whether to continue or halt.
        fn inference_callback<'a>(
            stop_sequence: String, // Takes ownership of the stop sequence.
            buf: &'a mut String, // A mutable reference to a buffer string, allowing modification.
            tx: tokio::sync::mpsc::Sender<String>, // A transmitter for sending String messages asynchronously.
            runtime: &'a mut tokio::runtime::Runtime // A mutable reference to the Tokio runtime.
        ) -> impl (FnMut(llm::InferenceResponse) -> Result<llm::InferenceFeedback, Infallible>) +
            'a {
            // Importing specific feedback types for convenience.
            use llm::InferenceFeedback::Halt;
            use llm::InferenceFeedback::Continue;

            // The `move` keyword captures the variables by value, making them part of the closure's environment.
            move |resp| -> Result<llm::InferenceFeedback, Infallible> {
                match resp {
                    // Matches on the type of inference response received.
                    llm::InferenceResponse::InferredToken(t) => {
                        // Cloning `buf` to create a temporary string for comparison without altering `buf`.
                        let mut reverse_buf = buf.clone();
                        reverse_buf.push_str(t.as_str()); // Appends the new token to the temporary string.
                        if stop_sequence.as_str().eq(reverse_buf.as_str()) {
                            // If the stop sequence matches exactly, clear the buffer and halt.
                            buf.clear();
                            return Ok(Halt);
                        } else if stop_sequence.as_str().starts_with(reverse_buf.as_str()) {
                            // If the stop sequence starts with the temporary buffer, append the token to `buf`.
                            buf.push_str(t.as_str());
                            return Ok(Continue);
                        }

                        // Prepares the message to send, based on whether `buf` is empty.
                        let text_to_send = if buf.is_empty() { t.clone() } else { reverse_buf };

                        // Clones the transmitter to allow sending from within the async block.
                        let tx_cloned = tx.clone();
                        // Executes an async block using the runtime, sending the message.
                        runtime.block_on(async move {
                            tx_cloned.send(text_to_send).await.expect("issue sending on channel");
                        });

                        Ok(Continue) // Continues the inference process.
                    }
                    llm::InferenceResponse::EotToken => Ok(Halt), // Halts on an End of Transmission token.
                    _ => Ok(Continue), // Continues for any other response type.
                }
            }
        }

        /// Handles WebSocket connections, setting up channels for message transmission and processing incoming messages.
        ///
        /// # Arguments
        /// * `req` - The incoming HTTP request to upgrade to a WebSocket.
        /// * `body` - The request payload.
        /// * `model` - A thread-safe reference to the model, wrapped for sharing across async tasks.
        ///
        /// # Returns
        /// A result containing the HTTP response to initiate the WebSocket connection or an error.
        pub async fn ws(
            req: HttpRequest,
            body: Payload,
            model: web::Data<Llama>
        ) -> Result<HttpResponse, Error> {
            // Initiates the WebSocket handshake and captures the response, session, and message stream.
            let (response, session, mut msg_stream) = actix_ws::handle(&req, body)?;

            // Mutex for thread-safe access and mpsc for message passing in async contexts.
            use std::sync::Mutex;
            use tokio::sync::mpsc;

            // Sets up a channel for sending inference results with a buffer size of 100 messages.
            let (send_inference, mut receive_inference) = mpsc::channel(100);

            // Clones the model reference for use across threads, ensuring thread safety.
            let mdl: Arc<Llama> = model.into_inner().clone();
            // Wraps the session in an `Arc<Mutex>` for shared, thread-safe access.
            let sess = Arc::new(Mutex::new(session));
            let sess_cloned = sess.clone();

            // Spawns an async task for receiving WebSocket messages and processing them.
            actix_rt::spawn(async move {
                // Sets up a standard synchronous channel for new user messages.
                let (send_new_user_message, receive_new_user_message) = std::sync::mpsc::channel();

                // Spawns a separate thread for handling inference, to keep it on the same thread due to library limitations.
                std::thread::spawn(move || {
                    // Sets up a new inference session with the cloned model.
                    let mut inference_session = session_setup(mdl);

                    // Processes each new user message received, performing inference and sending results.
                    for new_user_message in receive_new_user_message {
                        let _ = infer(
                            model_cloned.clone(),
                            &mut inference_session,
                            &new_user_message,
                            send_inference.clone()
                        );
                    }
                });

                // Processes each message received over the WebSocket.
                while let Some(Ok(msg)) = msg_stream.next().await {
                    match msg {
                        // Responds to ping messages to keep the connection alive.
                        Msg::Ping(bytes) => {
                            let res = sess_cloned.lock().unwrap().pong(&bytes).await;
                            if res.is_err() {
                                return;
                            }
                        }
                        // Sends text messages to the inference thread for processing.
                        Msg::Text(s) => {
                            let _ = send_new_user_message.send(s.to_string());
                        }
                        // Breaks the loop for any other message type, closing the connection.
                        _ => {
                            break;
                        }
                    }
                }
            });

            // Spawns another async task for sending inference results over the WebSocket.
            actix_rt::spawn(async move {
                while let Some(message) = receive_inference.recv().await {
                    sess.lock().unwrap().text(message).await.expect("issue sending on websocket");
                }
            });

            Ok(response) // Returns the response to finalize the WebSocket handshake.
        }
    }
}
