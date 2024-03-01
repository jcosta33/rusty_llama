// The `use` statement in Rust is similar to `import` in JavaScript. It's used to bring libraries or modules into scope for use in the current file.
use cfg_if::cfg_if;

// Uses a conditional compilation block to include code based on whether the "ssr" (server-side rendering) feature is enabled.
// In Rust, conditional compilation is managed at compile-time, allowing for more optimized builds depending on the features enabled.
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
        use actix_web::HttpRequest; // Represents client requests to the server.
        use actix_web::HttpResponse; // For constructing server responses to send back to clients.
        use actix_web::web::Payload; // The payload of a request, such as form data or JSON.
        use actix_web::Error; // General error type for handling issues in request processing.
        use actix_ws::Message as Msg; // WebSocket message handling, with `as` renaming it for clarity.
        use futures::stream::StreamExt; // Extensions for working with asynchronous streams.
        use leptos::*; // Assuming `leptos` is a framework or utility library, importing everything from it.
        use tokio::sync::mpsc;
        use llm::{
            KnownModel, // A trait indicating the model's capabilities.
            InferenceSession,
            InferenceRequest,
            InferenceParameters,
            InferenceFeedback,
            InferenceResponse,
            feed_prompt_callback,
        };
        use tokio::runtime::Runtime;

        /// Performs AI model inference based on a user's message and sends the result over a WebSocket connection.
        ///
        /// # Arguments
        /// * `model` - A shared, thread-safe reference to the AI model.
        /// * `session` - A mutable reference to the inference session, allowing the session to be updated.
        /// * `user_message` - The message from the user, wrapped in a `String`.
        /// * `tx` - The transmitter part of a channel for sending the inference result back to the client.
        ///
        /// # Returns
        /// This function returns a `Result` type, which is either `Ok(())` indicating success, or `Err(ServerFnError)` indicating an error occurred.
        pub fn perform_inference(
            model: Arc<Llama>,
            session: &mut llm::InferenceSession,
            user_message: &String,
            sender: mpsc::Sender<String>
        ) -> Result<(), ServerFnError> {
            // Bringing the Tokio runtime into scope specifically for this function. Required for running asynchronous tasks synchronously.
            use tokio::runtime::Runtime;

            // Creates a new Tokio runtime, panicking if this fails. A runtime is required for executing asynchronous code.
            // `expect` is a method that panics with the specified message if the called function returns an `Err`.
            let mut async_runtime = Runtime::new().expect("issue creating tokio runtime");

            // Performs the inference by calling the `infer` method on the session, passing in the model, a random number generator,
            // and the inference request constructed from the user's message.
            // `.unwrap_or_else(|e| panic!("{e}"))` will panic if the inference fails, providing the error message.
            session
                .infer(
                    model.as_ref(), // Converts `Arc<Llama>` into a standard reference `&Llama`, allowing the model to be used without taking ownership.
                    &mut rand::thread_rng(), // Creates a thread-local random number generator for the inference.
                    &(InferenceRequest { // Constructs the inference request, using `format!` to interpolate user and character names into the prompt.
                        prompt: format!("{USER_NAME}\n{user_message}\n{ASSISTANT_NAME}:")
                            .as_str() // Converts the formatted string into a string slice (`&str`).
                            .into(), // Converts the string slice into a type compatible with the inference request.
                        parameters: &InferenceParameters::default(), // Uses default parameters for the inference.
                        play_back_previous_tokens: false, // Indicates whether to play back previous tokens during inference.
                        maximum_token_count: None, // No maximum token count is specified, allowing the model to decide.
                    }),
                    &mut Default::default(), // Uses a default empty state for any additional required state.
                    process_inference_result(
                        String::from(USER_NAME),
                        &mut String::new(),
                        sender,
                        &mut async_runtime
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
        fn setup_session(model: Arc<Llama>) -> InferenceSession {
            // Starts a new session with the AI model, using default settings.
            let mut session = model.start_session(Default::default());
            // Feeds the initial prompt or context into the session, preparing it for further inferences.
            session
                .feed_prompt(
                    model.as_ref(), // As before, converts `Arc<Llama>` to `&Llama`.
                    "A chat between a human and an assistant.",
                    &mut Default::default(), // Uses default settings for any additional state.
                    feed_prompt_callback(|_| {
                        // Defines a callback for handling the AI's response to the initial prompt.
                        Ok::<InferenceFeedback, Infallible>(InferenceFeedback::Continue)
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
        /// * `sender` - A transmitter for sending messages over a channel.
        /// * `runtime` - The Tokio runtime for executing asynchronous tasks.
        ///
        /// # Returns
        /// A closure that processes inference responses and decides whether to continue or halt.
        fn process_inference_result<'a>(
            stop_sequence: String, // Takes ownership of the stop sequence.
            buf: &'a mut String, // A mutable reference to a buffer string, allowing modification.
            sender: mpsc::Sender<String>, // A transmitter for sending String messages asynchronously.
            runtime: &'a mut Runtime // A mutable reference to the Tokio runtime.
        ) -> impl // In Rust, impl is a keyword that defines an implementation block for a trait or type. In this case, it's used to define a closure that implements the FnMut trait.

        // FnMut is a trait for mutable function pointers, allowing the closure to be called and modified. Without this trait, the closure would be immutable.
        // The + sign indicates that the closure implements multiple traits, in this case FnMut and 'a, which is a lifetime specifier.
        (FnMut(InferenceResponse) -> Result<llm::InferenceFeedback, Infallible>) +
            'a {
            // Importing specific feedback types for convenience.
            use llm::InferenceFeedback::Halt;
            use llm::InferenceFeedback::Continue;

            // The `move` keyword captures the variables by value, making them part of the closure's environment.
            move |response| -> Result<llm::InferenceFeedback, Infallible> {
                // In Rust, match is used for pattern matching, similar to a switch statement in JavaScript.
                match response {
                    // Matches on the type of inference response received. If it's an inferred token, it processes the token.
                    InferenceResponse::InferredToken(token) => {
                        // Cloning `buf` to create a temporary string for comparison without altering `buf`.
                        // A buffer is used to accumulate tokens until the stop sequence is reached.
                        // This is necessary because the AI model may return tokens in multiple responses.
                        // We use clone here to avoid moving the buffer, which would invalidate the reference.
                        let mut temp_buffer = buf.clone();
                        temp_buffer.push_str(token.as_str()); // Appends the new token to the temporary string.
                        if stop_sequence.as_str().eq(temp_buffer.as_str()) {
                            // If the stop sequence matches exactly, clear the buffer and halt.
                            // With the buffer clear, the next token will be sent as a new message.
                            buf.clear();
                            return Ok(Halt);
                        } else if
                            // If the stop sequence starts with the temporary buffer, append the token to `buf`.
                            // We do this to accumulate tokens until the stop sequence is reached.
                            stop_sequence.starts_with(temp_buffer.as_str())
                        {
                            // t is a token, so we can safely unwrap it here.
                            buf.push_str(token.as_str());
                            return Ok(Continue);
                        }

                        // Prepares the message to send, based on whether `buf` is empty.
                        // In Rust we can use the `if` expression to return different values based on a condition.
                        let text_to_send = if buf.is_empty() { token.clone() } else { temp_buffer };

                        // Clones the transmitter to allow sending from within the async block.
                        let sender_cloned = sender.clone();
                        // Executes an async block using the runtime, sending the message.
                        runtime.block_on(async {
                            sender_cloned
                                .send(text_to_send).await
                                .expect("issue sending on channel");
                        });

                        Ok(Continue) // Continues the inference process.
                    }
                    // If the response is an end of transmission token, it halts the inference process.
                    InferenceResponse::EotToken => Ok(Halt), // Halts on an End of Transmission token.
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
                // let send_inference_cloned = send_inference.clone();
                // Rustformers sessions need to stay on the same thread
                // So we can't really rely on TOKIOOOOO
                let model_cloned = mdl.clone();
                // Spawns a separate thread for handling inference, to keep it on the same thread due to library limitations.
                std::thread::spawn(move || {
                    // Sets up a new inference session with the cloned model.
                    let mut session = setup_session(mdl);

                    // Processes each new user message received, performing inference and sending results.
                    for new_user_message in receive_new_user_message {
                        let _ = perform_inference(
                            model_cloned.clone(),
                            &mut session,
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
