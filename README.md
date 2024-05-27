## QuizQueue

QuizQueue is a platform that creates MCQ quizzes for user's documents.

## Installation

Follow step-by-step instructions to install the necessary components.

1. **Clone the repository:**

    ```sh
    git clone https://github.com/QuizQueue/QuizQueue.git
    
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```sh
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

Explain how to run your code. Include examples of the commands that should be used in the terminal.

1. **Setting variables:**
   ```sh
   export OPENAI_API_KEY='your_api_goes_here'
   export HUGGINGFACEHUB_API_TOKEN='your_hf_token'
   ```
2. **Verifiying keys:**

    If your script requires arguments, provide examples:

    ```sh
    echo $OPENAI_API_KEY
    echo $HUGGINGFACEHUB_API_TOKEN
    ```

3. **Running the script:**

    ```sh
    chainlit run quizQueue.py
    ```


## Contributing

Provide guidelines for contributing to your project, if applicable.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

Specify the license under which the project is distributed.

