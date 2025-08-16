import numpy as np
import re
import itertools

class TBSR:

    def __init__(self, vocab_size, label_map, tensor_dim=4, learning_rate=0.01):
        
        self.vocab_size = vocab_size
        self.label_map = label_map
        self.num_classes = len(label_map)
        self.tensor_dim = tensor_dim
        self.lr = learning_rate

        self.word_tensors = np.random.randn(vocab_size, tensor_dim, tensor_dim, tensor_dim) * 0.1

        self.W_out = np.random.randn(tensor_dim, self.num_classes) * 0.1
        self.b_out = np.zeros(self.num_classes)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    def forward(self, sentence_indices):
        tensors = [self.word_tensors[i] for i in sentence_indices]

        if not tensors:
            return np.zeros(self.num_classes), {}
        
        resonant_tensor = tensors[0]
        intermediate_tensors = [resonant_tensor]

        for i in range(1, len(tensors)):
            resonant_tensor = np.tensordot(resonant_tensor, tensors[i], axes=([1,2],[0,1]))
            intermediate_tensors.append(resonant_tensor)


        output_scores = np.dot(resonant_tensor, self.W_out) + self.b_out
        final_output = self._softmax(output_scores)

        cache = {
            "tensors": tensors,
            "intermediate_tensors": intermediate_tensors,
            "resonant_tensor": resonant_tensor,
            "output_scores": output_scores,
        }

        return final_output, cache

    def backward(self, sentence_indices, cache, y_true):
        tensors = cache["tensors"]
        resonant_tensor = cache["resonant_tensor"]
        output_scores = cache["output_scores"]

        grad_output = self._softmax(output_scores) - y_true

        grad_W_out = np.outer(resonant_tensor, grad_output)
        grad_b_out = grad_output

        grad_resonant_tensor = np.dot(grad_output, self.W_out.T)

        grad_tensors = [np.zeros_like(t) for t in tensors]

        for i in range(len(tensors) - 1, 0, -1):
            prev_resonant_tensor = cache["intermediate_tensors"][i-1]
            current_tensor = tensors[i]

            grad_tensors[i] += np.tensordot(prev_resonant_tensor, grad_resonant_tensor, axes=([0],[0]))

            grad_resonant_tensor = np.tensordot(grad_resonant_tensor, current_tensor.T, axes=([0],[0]))


        grad_tensors[0] += grad_resonant_tensor

        self.W_out -= self.lr * grad_W_out
        self.b_out -= self.lr * grad_b_out
        for i, idx in enumerate(sentence_indices):
            self.word_tensors[idx] -= self.lr * grad_tensors[i]

    def train(self, data, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            for sentence_indices, label_index in data:
                if not sentence_indices:
                    continue

                y_true = np.zeros(self.num_classes)
                y_true[label_index] = 1

                y_pred, cache = self.forward(sentence_indices)

                loss = -np.sum(y_true * np.log(y_pred + 1e-9))
                total_loss += loss

                self.backward(sentence_indices, cache, y_true)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data)}")

    def predict(self, sentence, word_to_idx):
        tokens = simple_tokenizer(sentence)
        sentence_indices = [word_to_idx.get(word, 0) for word in tokens]

        # Forward pass
        scores, _ = self.forward(sentence_indices)
        predicted_index = np.argmax(scores)

        # Reverse map index to label
        idx_to_label = {v: k for k, v in self.label_map.items()}
        return idx_to_label[predicted_index]


def simple_tokenizer(text):
    """A simple tokenizer that converts text to lowercase and splits by non-alphanumeric characters."""
    return [word for word in re.split(r'\W+', text.lower()) if word]


def load_and_preprocess_data(file_contents_dict):
    all_words = []
    all_labels = set()
    data = []

    for filename, content in file_contents_dict.items():
        lines = content.strip().split('\n')[1:] # Skip header
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text, label = parts
                tokens = simple_tokenizer(text)
                all_words.extend(tokens)
                all_labels.add(label)
                data.append((tokens, label))

    # Create vocabulary and label map
    word_counts = {word: all_words.count(word) for word in set(all_words)}
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    
    # Let's limit the vocab size for this example to keep it manageable
    vocab_size = 2000
    vocab = sorted_words[:vocab_size-1] # -1 for the UNK token
    
    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx["<UNK>"] = 0 # Unknown word token
    
    label_map = {label: i for i, label in enumerate(all_labels)}

    # Convert data to indices
    processed_data = []
    for tokens, label in data:
        sentence_indices = [word_to_idx.get(token, 0) for token in tokens]
        label_index = label_map[label]
        processed_data.append((sentence_indices, label_index))

    return processed_data, word_to_idx, label_map


if __name__ == "__main__":
    # --- This is where you would place the file contents you fetched ---
    # For this example, I'm using a small, representative sample of the data.
    # In a real scenario, you would pass the full file contents.
    file_contents = {
        "airline-sentiment.tsv": """text	label
@VirginAmerica What @dhepburn said.	neutral
@VirginAmerica plus you've added commercials to the experience... tacky.	positive
@VirginAmerica I didn't today... Must mean I need to take another trip!	neutral
"@VirginAmerica it's really aggressive to blast obnoxious ""entertainment"" in your guests' faces & they have little recourse"	negative
@VirginAmerica and it's a really big bad thing about it	negative
"""
    }

    # 1. Load and process the data
    training_data, word_to_idx, label_map = load_and_preprocess_data(file_contents)
    vocab_size = len(word_to_idx)

    # 2. Initialize the model
    print("Initializing TBSR model...")
    tbsr_model = TBSR(vocab_size=vocab_size, label_map=label_map, tensor_dim=3, learning_rate=0.01)

    # 3. Train the model
    print("Starting training...")
    tbsr_model.train(training_data, epochs=25)
    print("Training complete.")

    # 4. Make some predictions
    print("\n--- Predictions ---")
    test_sentence_1 = "The airline has a really big bad thing."
    prediction_1 = tbsr_model.predict(test_sentence_1, word_to_idx)
    print(f"Sentence: '{test_sentence_1}'")
    print(f"Predicted Sentiment: {prediction_1}")

    test_sentence_2 = "What a great experience."
    prediction_2 = tbsr_model.predict(test_sentence_2, word_to_idx)
    print(f"Sentence: '{test_sentence_2}'")
    print(f"Predicted Sentiment: {prediction_2}")