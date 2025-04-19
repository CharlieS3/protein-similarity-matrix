import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import ast
import matplotlib.pyplot as plt
from transformers import T5Model, T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModelForSeq2SeqLM

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cos = torch.nn.CosineSimilarity(dim=0)

class ProteinSimilarityCNN:
    def __init__(self, input_shape=(None, None, 1), min_size=4):
        self.min_size = min_size
        self.model = self._build_model(input_shape)


    def _build_model(self, input_shape):
        #input layer with dynamic shape to handle variable-sized similarity matrices
        inputs = layers.Input(shape=input_shape)
        
        #first convolutional block
        #32 filters means we have 32 different 3×3 filters in this convolutional layer
        #padding='same' ensures output has the same dimensions as input (for convolutional layer)
        #the 3x3 filter dot products with a 3x3 peice of them matrix giving a single value that ReLU is applied to
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

        #subtract the mean, scale (divide by std) and shift/scaling
        x = layers.BatchNormalization()(x)

        #takes a 2×2 region in x and keeps only the max value
        #padding='same' ensures that if the input size is odd it doesn't shrink more than necessary by adding padding
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        ###############################################################################
        # if input was (H, W, 1) → After these three lines, it becomes (H/2, W/2, 32) #
        ###############################################################################
        
        #second convolutional block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        #third convolutional block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)

        #####################################
        # x's size is now (H/4, W/4, 128) #
        #####################################
        
        #multi-scale region detection - to capture different sized regions
        #these layers will detect region corners at different scales
        corner_detection = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        
        #global Average Pooling to reduce to a fixed size
        #gap computes the average of each feature map reducing it to a single scalar value per feature (channel)
        gap = layers.GlobalAveragePooling2D()(corner_detection)

        #we now have a 1D tensor of shape: (batch_size,64)

        #learns 128 new features from gap by applying weighted transformations (matrix mult) and non-linearity (ReLU activation)
        dense = layers.Dense(128, activation='relu')(gap)
        
        #these are the predicted bounding box corners (x1, y1, x2, y2)normalized to [0,1] for each image in the batch
        outputs = layers.Dense(4, activation='sigmoid')(dense)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def compile_model(self, learning_rate=0.001):

        #the Adam optimizer updates model weights using both:
        #momentum (helps smooth updates, like a moving average of past gradients)
        #adaptive Learning Rates (adjusts step size for each parameter based on past updates)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        def ciou_loss(y_true, y_pred, epsilon=1e-6):
            #extract coordinates labels
            x1_true, y1_true, x2_true, y2_true = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
            x1_pred, y1_pred, x2_pred, y2_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]

            #calculate box widths and heights
            w_true = x2_true - x1_true
            h_true = y2_true - y1_true
            w_pred = x2_pred - x1_pred
            h_pred = y2_pred - y1_pred

            #clamp predicted width/height to avoid issues with negative values during training
            w_pred = tf.maximum(w_pred, epsilon)
            h_pred = tf.maximum(h_pred, epsilon)

            #calculate areas
            area_true = w_true * h_true
            area_pred = w_pred * h_pred

            #calculate intersection area
            x_overlap = tf.maximum(0.0, tf.minimum(x2_true, x2_pred) - tf.maximum(x1_true, x1_pred))
            y_overlap = tf.maximum(0.0, tf.minimum(y2_true, y2_pred) - tf.maximum(y1_true, y1_pred))
            intersection = x_overlap * y_overlap

            #calculate union area
            union = area_true + area_pred - intersection

            #calculate IoU
            iou = intersection / (union + epsilon)

            #calculate CIoU Penalties

            # 1. Center distance penalty
            center_x_true = (x1_true + x2_true) / 2.0
            center_y_true = (y1_true + y2_true) / 2.0
            center_x_pred = (x1_pred + x2_pred) / 2.0
            center_y_pred = (y1_pred + y2_pred) / 2.0

            center_dist_sq = tf.square(center_x_true - center_x_pred) + tf.square(center_y_true - center_y_pred)

            #smallest enclosing box diagonal squared
            enclose_x1 = tf.minimum(x1_true, x1_pred)
            enclose_y1 = tf.minimum(y1_true, y1_pred)
            enclose_x2 = tf.maximum(x2_true, x2_pred)
            enclose_y2 = tf.maximum(y2_true, y2_pred)
            enclose_diag_sq = tf.square(enclose_x2 - enclose_x1) + tf.square(enclose_y2 - enclose_y1)

            #distance penalty term (DIoU term)
            distance_penalty = center_dist_sq / (enclose_diag_sq + epsilon)

            # 2. Aspect ratio penalty
            arctan_true = tf.atan(w_true / (h_true + epsilon))
            arctan_pred = tf.atan(w_pred / (h_pred + epsilon))
            v = (4.0 / (np.pi ** 2)) * tf.square(arctan_true - arctan_pred)

            #trade-off parameter alpha (needs positive IoU)
            #stop gradient for alpha calculation to treat it as a constant
            alpha = tf.stop_gradient(v / (1.0 - iou + v + epsilon))

            #aspect ratio penalty term
            aspect_ratio_penalty = alpha * v

            #calculate Final CIoU Loss 
            ciou = iou - distance_penalty - aspect_ratio_penalty
            ciou_loss = 1.0 - ciou

            #average loss over the batch
            return tf.reduce_mean(ciou_loss)

        def region_loss_ciou_l1(y_true, y_pred, weight_l1=1.0, weight_ciou=1.0):
            # L1 Loss on coordinates
            l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

            # CIoU Loss
            loss_ciou = ciou_loss(y_true, y_pred)

            # Combined loss
            return weight_l1 * l1_loss + weight_ciou * loss_ciou
        
        self.model.compile(optimizer=optimizer, loss=region_loss_ciou_l1, metrics=['mae'])
        return self.model
    
    def prepare_data(self, similarity_matrices, region_indices):
        X_processed = []
        y_processed = []
        
        for matrix, indices in zip(similarity_matrices, region_indices):
            #normalize matrix to [0, 1] range
            matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix) + 1e-6)
            
            #adds a channel dimension converting a (H, W) matrix to (H, W, 1)
            matrix = np.expand_dims(matrix, axis=-1)
            X_processed.append(matrix)
            
            #extract coordinates
            (i1, j1), (i2, j2), (i3, j3), (i4, j4) = indices

            #store height and width
            h, w = matrix.shape[:2]
            
            #normalize coordinates to [0, 1]
            x1, y1 = j1 / w, i1 / h 
            x2, y2 = j4 / w, i4 / h 
            
            #traget is now a single vector
            target = np.array([x1, y1, x2, y2])
            y_processed.append(target)
        
        return X_processed, np.array(y_processed)
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=8, epochs=10, patience=20):
        #custom training loop to handle variable-sized inputs
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        history = {'loss': [], 'val_loss': []} # stores training and validation loss for plotting
        
        for epoch in range(epochs):
            #shuffle training data
            indices = np.random.permutation(len(X_train))
            epoch_loss = 0
            
            #######################################################################################
            # Within each batch each example is trained individually because the input sizes vary #
            #######################################################################################

            #train on batches
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_loss = 0
                
                #Since input sizes vary this processes each sample individually
                for idx in batch_indices:
                    x = np.expand_dims(X_train[idx], axis=0)  # Add batch dimension
                    y = np.expand_dims(y_train[idx], axis=0)  # Add batch dimension
                    
                    # Train on this single sample
                    loss = self.model.train_on_batch(x, y)[0]
                    batch_loss += loss
                
                #average batch loss
                batch_loss /= len(batch_indices)
                epoch_loss += batch_loss * len(batch_indices)
            
            #average epoch loss
            epoch_loss /= len(X_train)
            history['loss'].append(epoch_loss)
            
            #validate
            val_loss = 0
            for i in range(len(X_val)):
                x = np.expand_dims(X_val[i], axis=0)
                y = np.expand_dims(y_val[i], axis=0)
                val_loss += self.model.evaluate(x, y, verbose=0)[0]
            
            val_loss /= len(X_val)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
            
            #check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_weights('best_protein_similarity_cnn.h5.weights.h5')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            #early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        #load best weights
        self.model.load_weights('best_protein_similarity_cnn.h5.weights.h5')
        
        return history
    
    def predict_region(self, similarity_matrix):
        #normalize matrix (1e-6 to avoid possible division by zero)
        matrix = (similarity_matrix - np.min(similarity_matrix)) / (np.max(similarity_matrix) - np.min(similarity_matrix) + 1e-6)
        
        #add batch and channel dimensions (1, H, W, 1)
        matrix = np.expand_dims(np.expand_dims(matrix, axis=0), axis=-1)
        
        #called trained model to predict coordinates [x1, y1, x2, y2] 
        #[0] removes the batch dimension (it gives [0.12, 0.34, 0.56, 0.78] instead of [[0.12, 0.34, 0.56, 0.78]])
        #this what VALUES in the matrix we think are the corners
        predicted_coords = self.model.predict(matrix)[0]
        
        #extract height and width as was as the coordinates
        h, w = similarity_matrix.shape
        x1, y1, x2, y2 = predicted_coords
        
        #convert the values we think are the corners to actaul coordinates
        i1, j1 = int(y1 * h), int(x1 * w)  # Top-left
        i4, j4 = int(y2 * h), int(x2 * w)  # Bottom-right
        i2, j2 = i1, j4  # Top-right
        i3, j3 = i4, j1  # Bottom-left
        
        #ensure coordinates are within bounds
        i1 = max(0, min(i1, h-1))
        j1 = max(0, min(j1, w-1))
        i2 = max(0, min(i2, h-1))
        j2 = max(0, min(j2, w-1))
        i3 = max(0, min(i3, h-1))
        j3 = max(0, min(j3, w-1))
        i4 = max(0, min(i4, h-1))
        j4 = max(0, min(j4, w-1))
        
        #ensure minimum size
        width = j2 - j1
        height = i3 - i1

        #expands symmetrically left & right.
        if width < self.min_size:
            expand = (self.min_size - width) // 2
            j1 = max(0, j1 - expand)
            j4 = min(w-1, j4 + expand)
            j2, j3 = j4, j1  #update other corners

        #expands symmetrically up & down.
        if height < self.min_size:
            expand = (self.min_size - height) // 2
            i1 = max(0, i1 - expand)
            i3 = min(h-1, i3 + expand)
            i2, i4 = i1, i3  #update other corners
        
        #return corners
        return [(i1, j1), (i2, j2), (i3, j3), (i4, j4)]
    
    def visualize_prediction(self, similarity_matrix, predicted_region, true_region=None):
        plt.figure(figsize=(6, 4))
        plt.imshow(similarity_matrix, cmap='coolwarm')
        
        #plot predicted region
        (i1, j1), (i2, j2), (i3, j3), (i4, j4) = predicted_region
        width = j2 - j1
        height = i3 - i1
        rect = plt.Rectangle((j1, i1), width, height, edgecolor='blue', facecolor='none', linewidth=2, label='Predicted')
        plt.gca().add_patch(rect)
        
        #plot true region if provided
        if true_region:
            (i1, j1), (i2, j2), (i3, j3), (i4, j4) = true_region
            width = j2 - j1
            height = i3 - i1
            rect = plt.Rectangle((j1, i1), width, height, edgecolor='lime', facecolor='none', linewidth=2, label='Ground Truth')
            plt.gca().add_patch(rect)
        
        plt.colorbar(label='Similarity')
        plt.title('Similarity Matrix with Predicted Region')
        plt.legend()
        plt.tight_layout()
        plt.show()

def calculate_overlap_region(pred_region, true_region):
    # Extract coordinates
    (i1_pred, j1_pred), (i2_pred, j2_pred), (i3_pred, j3_pred), (i4_pred, j4_pred) = pred_region
    (i1_true, j1_true), (i2_true, j2_true), (i3_true, j3_true), (i4_true, j4_true) = true_region
    
    # Convert to top-left and bottom-right coordinates
    x1_pred, y1_pred = min(j1_pred, j3_pred), min(i1_pred, i2_pred)
    x2_pred, y2_pred = max(j2_pred, j4_pred), max(i3_pred, i4_pred)
    
    x1_true, y1_true = min(j1_true, j3_true), min(i1_true, i2_true)
    x2_true, y2_true = max(j2_true, j4_true), max(i3_true, i4_true)
    
    # Calculate intersection coordinates
    overlap_x1 = max(x1_pred, x1_true)
    overlap_y1 = max(y1_pred, y1_true)
    overlap_x2 = min(x2_pred, x2_true)
    overlap_y2 = min(y2_pred, y2_true)
    
    # Check if there is an overlap
    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
        overlap_width = overlap_x2 - overlap_x1
        overlap_height = overlap_y2 - overlap_y1
        overlap_coords = (overlap_y1, overlap_x1, overlap_width, overlap_height)
    else:
        overlap_coords = None
    
    # Calculate intersection area
    x_overlap = max(0, overlap_x2 - overlap_x1)
    y_overlap = max(0, overlap_y2 - overlap_y1)
    intersection = x_overlap * y_overlap
    
    # Calculate individual areas
    area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    area_true = (x2_true - x1_true) * (y2_true - y1_true)
    
    union = area_pred + area_true - intersection # Calculate union area
    iou = intersection / union if union > 0 else 0.0
    
    return overlap_coords, iou

def get_unique_filename(base_name):
    name, ext = os.path.splitext(base_name)
    counter = 1
    unique_name = base_name
    while os.path.exists(unique_name):
        unique_name = f"{name}_{counter}{ext}"
        counter += 1
    return unique_name

def visualize_prediction_with_overlap(similarity_matrix, predicted_region, true_region, save_name="prediction_visualization.png"):
    # Calculate overlap region and IoU
    overlap_coords, iou = calculate_overlap_region(predicted_region, true_region)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='coolwarm')
    
    # Plot predicted region
    (i1, j1), (i2, j2), (i3, j3), (i4, j4) = predicted_region
    width = j2 - j1
    height = i3 - i1
    rect_pred = plt.Rectangle((j1, i1), width, height, edgecolor='blue', facecolor='none', linewidth=2, label='Predicted')
    plt.gca().add_patch(rect_pred)
    
    # Plot true region
    (i1, j1), (i2, j2), (i3, j3), (i4, j4) = true_region
    width = j2 - j1
    height = i3 - i1
    rect_true = plt.Rectangle((j1, i1), width, height, edgecolor='lime', facecolor='none', linewidth=2, label='Ground Truth')
    plt.gca().add_patch(rect_true)
    
    # Plot overlap region if one exists
    if overlap_coords:
        top_left_i, top_left_j, width, height = overlap_coords
        rect_overlap = plt.Rectangle((top_left_j, top_left_i), width, height, 
                                     edgecolor='darkred', facecolor='darkred', alpha=0.5, 
                                     linewidth=1, label='Overlap')
        plt.gca().add_patch(rect_overlap)
    
    plt.colorbar(label='Similarity')
    plt.title(f'Similarity Matrix with Regions - IoU: {iou:.4f} ({iou*100:.2f}%)')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure with a unique name
    unique_filename = get_unique_filename(save_name)
    plt.savefig(unique_filename, dpi=300)
    plt.close()
    
    return iou

def plot_test_iou_vs_train_size(model_class, similarity_matrices, region_indices, train_indices, val_indices,interval=100, visualize_samples=False):
    train_sizes = list(range(interval, len(train_indices) + 1, interval))
    val_ious = []

    for size in train_sizes:
        print(f"\n--- Training with {size} samples ---")
        torch.cuda.empty_cache()

        # Subset the training data
        subset_indices = train_indices[:size]
        X_train = [similarity_matrices[i] for i in subset_indices]
        y_train = [region_indices[i] for i in subset_indices]
        X_val = [similarity_matrices[i] for i in val_indices]
        y_val = [region_indices[i] for i in val_indices]

        # Initialize and compile new model
        model = model_class(input_shape=(None, None, 1))
        model.compile_model()

        # Prepare data
        X_train_prep, y_train_prep = model.prepare_data(X_train, y_train)
        X_val_prep, y_val_prep = model.prepare_data(X_val, y_val)

        # Train the model
        model.train(X_train_prep, y_train_prep, X_val_prep, y_val_prep,
                    batch_size=4, epochs=10, patience=3)

        # Evaluate on validation set
        iou_scores = []
        for i in range(len(X_val)):
            pred = model.predict_region(X_val[i])
            _, iou = calculate_overlap_region(pred, y_val[i])
            iou_scores.append(iou)
        avg_iou = sum(iou_scores) / len(iou_scores)
        val_ious.append(avg_iou)
        print(f"Avg IoU with {size} samples: {avg_iou:.4f}")

    # Plot only IoU vs. training size
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, val_ious, marker='o', color='blue')
    plt.title("Test IoU vs Training Set Size")
    plt.xlabel("Training Set Size")
    plt.ylabel("Test IoU")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_iou_vs_training_size.png", dpi=300)
    plt.close()

def main():
    num = 10000
    similarity_matrices = np.load("similarity_matrices.npy", allow_pickle=True)
    similarity_matrices_list = list(similarity_matrices) 


    similarity_matrices_list = similarity_matrices_list[:num]

    region_indices = []
    with open("label_indices.txt", 'r') as f:
        for line in f:
            #remove any trailing newline or whitespace
            line = line.strip()
            if line:  # Make sure it's not an empty line
                #convert the string to a list of tuples
                region = ast.literal_eval(line)
                region_indices.append(region)

    region_indices = region_indices[:num]
    
    print(f"Total examples: {len(similarity_matrices_list)}")
    indices = np.arange(len(similarity_matrices_list))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

    #selects training samples from similarity_matrices and region_indices
    X_train = [similarity_matrices[i] for i in train_indices]
    y_train = [region_indices[i] for i in train_indices]

    #same for val and test set
    X_val = [similarity_matrices[i] for i in val_indices]
    y_val = [region_indices[i] for i in val_indices]
    X_test = [similarity_matrices[i] for i in test_indices]
    y_test = [region_indices[i] for i in test_indices]

    print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")

    input_shape = (None, None, 1)  # Dynamic input shape
    cnn_model = ProteinSimilarityCNN(input_shape)
    cnn_model.compile_model()

    X_train_prepared, y_train_prepared = cnn_model.prepare_data(X_train, y_train)
    X_val_prepared, y_val_prepared = cnn_model.prepare_data(X_val, y_val)

    plot_test_iou_vs_train_size(
    ProteinSimilarityCNN,
    similarity_matrices_list,
    region_indices,
    train_indices,
    val_indices,
    interval=200,
    visualize_samples=True)

if __name__ == "__main__":
    main()