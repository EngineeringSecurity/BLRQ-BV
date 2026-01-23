# T_SNE.py
"""
T-SNE visualization tool: Load T_resnet model and features, perform spherical fusion and visualize feature distributions by category
All visualizations are in English and scaled to fit within the frame
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import vector composing function
from Vector_composing import transform_vectors_numpy

# Import model class from T_resnet
from T_resnet import MultiModalResNet18

# Set English font and style with larger font sizes
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
sns.set_style("whitegrid")
sns.set_palette("husl")

class TSNEVisualizer:
    """TSNE Visualizer for multimodal feature analysis"""
    
    def __init__(self, model_path, config_path=None):
        """
        Initialize visualizer
        
        Parameters:
        model_path: Saved model path
        config_path: Configuration file path
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Load model
        self.model, self.model_config = self.load_model(model_path)
        
        # Set image preprocessing
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Feature storage
        self.features = {}
        self.labels = {}
        self.class_names = []
        
    def load_config(self, config_path):
        """Load configuration file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"Configuration loaded from {config_path}")
            return config
        else:
            # Default configuration
            config = {
                'feature_dim': 512,
                'class_names': ['non_rumor', 'rumor']
            }
            print("Using default configuration")
            return config
    
    def load_model(self, model_path):
        """Load T_resnet model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Try to get dimension information from checkpoint
        if 'config' in checkpoint:
            # Complete configuration saved in checkpoint
            config = checkpoint['config']
            npy1_dim = config.get('npy1_dim', 512)
            npy2_dim = config.get('npy2_dim', 4)
            feature_dim = config.get('feature_dim', 512)
            print(f"Loaded from checkpoint: npy1_dim={npy1_dim}, npy2_dim={npy2_dim}, feature_dim={feature_dim}")
        else:
            # Try to infer dimensions from model state dict
            model_state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Get linear layer weight dimensions
            npy1_fc_weight = model_state_dict.get('npy1_fc.weight', None)
            npy2_fc_weight = model_state_dict.get('npy2_fc.weight', None)
            
            if npy1_fc_weight is not None and npy2_fc_weight is not None:
                npy1_dim = npy1_fc_weight.shape[1]
                npy2_dim = npy2_fc_weight.shape[1]
                feature_dim = npy1_fc_weight.shape[0]  # Output dimension
                print(f"Inferred from weights: npy1_dim={npy1_dim}, npy2_dim={npy2_dim}, feature_dim={feature_dim}")
            else:
                # Use default values if inference fails
                npy1_dim = 4
                npy2_dim = 1024
                feature_dim = 512
                print(f"Using default dimensions: npy1_dim={npy1_dim}, npy2_dim={npy2_dim}, feature_dim={feature_dim}")
        
        # Create model
        model = MultiModalResNet18(
            npy1_dim=npy1_dim,
            npy2_dim=npy2_dim,
            num_classes=2,
            feature_dim=feature_dim
        ).to(self.device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        print(f"Model loaded from {model_path}")
        
        # Save configuration information
        model_config = {
            'npy1_dim': npy1_dim,
            'npy2_dim': npy2_dim,
            'feature_dim': feature_dim
        }
        
        return model, model_config
    
    def extract_features_from_data(self, image_dir, npy_dir1, npy_dir2, max_samples_per_class=200):
        """
        Extract features from dataset
        
        Parameters:
        image_dir: Image directory
        npy_dir1: Quantum feature directory (npy1)
        npy_dir2: BiLSTM feature directory (npy2)
        max_samples_per_class: Maximum samples per class (to prevent memory overflow)
        """
        print("\nStarting feature extraction...")
        
        # Get all classes
        if 'class_names' in self.config:
            class_names = self.config['class_names']
        else:
            # Get classes from directory
            class_names = [d for d in os.listdir(image_dir) 
                          if os.path.isdir(os.path.join(image_dir, d))]
        
        self.class_names = class_names
        print(f"Found classes: {class_names}")
        
        # Initialize storage
        self.features = {
            'image': [],
            'npy1': [],
            'npy2': [],
            'fused': [],
            'spherical_fused': []
        }
        self.labels = []
        self.sample_info = []
        
        # Get model expected dimensions
        npy1_dim = self.model_config['npy1_dim']
        npy2_dim = self.model_config['npy2_dim']
        
        # Process each class
        for class_idx, class_name in enumerate(class_names):
            print(f"\nProcessing class: {class_name} (index: {class_idx})")
            
            # Get class paths for images and features
            image_class_dir = os.path.join(image_dir, class_name)
            npy1_class_dir = os.path.join(npy_dir1, class_name)
            npy2_class_dir = os.path.join(npy_dir2, class_name)
            
            # Check if directories exist
            if not os.path.exists(image_class_dir):
                print(f"  Warning: Image directory does not exist: {image_class_dir}")
                continue
            if not os.path.exists(npy1_class_dir):
                print(f"  Warning: Quantum feature directory does not exist: {npy1_class_dir}")
                continue
            if not os.path.exists(npy2_class_dir):
                print(f"  Warning: BiLSTM feature directory does not exist: {npy2_class_dir}")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(image_class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit sample count
            if max_samples_per_class and len(image_files) > max_samples_per_class:
                import random
                random.seed(42)
                image_files = random.sample(image_files, max_samples_per_class)
                print(f"  Randomly selected {max_samples_per_class} samples")
            
            print(f"  Found {len(image_files)} image files")
            
            # Process each sample
            processed_count = 0
            for img_file in tqdm(image_files, desc=f"Extracting {class_name} features"):
                try:
                    # Build file paths
                    base_name = os.path.splitext(img_file)[0]
                    
                    image_path = os.path.join(image_class_dir, img_file)
                    npy1_path = os.path.join(npy1_class_dir, f"{base_name}.npy")
                    npy2_path = os.path.join(npy2_class_dir, f"{base_name}.npy")
                    
                    # Check if feature files exist
                    if not os.path.exists(npy1_path):
                        continue
                    if not os.path.exists(npy2_path):
                        continue
                    
                    # Load image
                    image = Image.open(image_path).convert('L')
                    image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    # Load npy features and adjust dimensions
                    npy1 = np.load(npy1_path).astype(np.float32).flatten()
                    npy2 = np.load(npy2_path).astype(np.float32).flatten()
                    
                    # Adjust dimensions to match model expectations
                    if len(npy1) > npy1_dim:
                        npy1 = npy1[:npy1_dim]
                    elif len(npy1) < npy1_dim:
                        npy1 = np.pad(npy1, (0, npy1_dim - len(npy1)), mode='constant')
                    
                    if len(npy2) > npy2_dim:
                        npy2 = npy2[:npy2_dim]
                    elif len(npy2) < npy2_dim:
                        npy2 = np.pad(npy2, (0, npy2_dim - len(npy2)), mode='constant')
                    
                    npy1_tensor = torch.tensor(npy1).unsqueeze(0).to(self.device)
                    npy2_tensor = torch.tensor(npy2).unsqueeze(0).to(self.device)
                    
                    # Extract features
                    with torch.no_grad():
                        outputs = self.model(image_tensor, npy1_tensor, npy2_tensor)
                        
                        # Get features based on T_resnet model forward return
                        if isinstance(outputs, tuple) and len(outputs) == 5:
                            # Model returns: output, image_features, npy1_features, npy2_features, fused_features
                            _, img_feat, npy1_feat, npy2_feat, fused_feat = outputs
                        else:
                            # For other output formats, adjust as needed
                            print(f"  Warning: Model output format unexpected: {type(outputs)}")
                            continue
                    
                    # Store original features (before mapping)
                    self.features['image'].append(img_feat.cpu().numpy().flatten())
                    self.features['npy1'].append(npy1_feat.cpu().numpy().flatten())
                    self.features['npy2'].append(npy2_feat.cpu().numpy().flatten())
                    self.features['fused'].append(fused_feat.cpu().numpy().flatten())
                    
                    # Perform spherical fusion
                    img_feat_np = img_feat.cpu().numpy().flatten()
                    npy1_feat_np = npy1_feat.cpu().numpy().flatten()
                    npy2_feat_np = npy2_feat.cpu().numpy().flatten()
                    
                    # Ensure three vectors have same dimension
                    min_dim = min(len(img_feat_np), len(npy1_feat_np), len(npy2_feat_np))
                    img_feat_np = img_feat_np[:min_dim]
                    npy1_feat_np = npy1_feat_np[:min_dim]
                    npy2_feat_np = npy2_feat_np[:min_dim]
                    
                    spherical_fused = transform_vectors_numpy(
                        img_feat_np, npy1_feat_np, npy2_feat_np, use_degrees=False
                    )
                    self.features['spherical_fused'].append(spherical_fused)
                    
                    # Store labels and information
                    self.labels.append(class_idx)
                    self.sample_info.append({
                        'image_path': image_path,
                        'base_name': base_name,
                        'class': class_name,
                        'class_idx': class_idx
                    })
                    
                    processed_count += 1
                    
                except Exception as e:
                    # Only print first few errors
                    if processed_count < 5:
                        print(f"    Error processing sample {img_file}: {e}")
                    continue
            
            print(f"  Successfully processed: {processed_count} samples")
        
        # Convert to numpy arrays
        for key in self.features:
            if self.features[key]:
                self.features[key] = np.array(self.features[key])
            else:
                self.features[key] = np.array([])
        
        self.labels = np.array(self.labels)
        
        print(f"\nFeature extraction completed:")
        print(f"  Total samples: {len(self.labels)}")
        print(f"  Samples per class:")
        for i, class_name in enumerate(self.class_names):
            count = np.sum(self.labels == i)
            print(f"    {class_name}: {count}")
        
        return self.features, self.labels
    
    def compute_tsne(self, features, n_components=2, perplexity=30, random_state=42):
        """
        Compute t-SNE dimensionality reduction
        
        Parameters:
        features: Feature matrix
        n_components: Reduced dimensions
        perplexity: t-SNE perplexity parameter
        random_state: Random seed
        
        Returns:
        t-SNE reduced results
        """
        print(f"\nComputing t-SNE reduction (perplexity={perplexity})...")
        
        # Check features
        if len(features) == 0:
            print("Warning: Feature array is empty")
            return None, None
        
        # If too many samples, random sampling
        max_samples = 5000
        if len(features) > max_samples:
            print(f"Many samples ({len(features)}), randomly sampling {max_samples}")
            indices = np.random.choice(len(features), max_samples, replace=False)
            features_subset = features[indices]
            labels_subset = self.labels[indices]
        else:
            features_subset = features
            labels_subset = self.labels
        
        # Compute t-SNE - adapt to different scikit-learn versions
        try:
            # Try new version parameters
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                max_iter=1000,
                learning_rate=200,
                init='pca'
            )
        except TypeError as e:
            # If parameter error, try old version
            if 'n_iter' in str(e) or 'max_iter' in str(e):
                try:
                    # Try n_iter parameter
                    tsne = TSNE(
                        n_components=n_components,
                        perplexity=perplexity,
                        random_state=random_state,
                        n_iter=1000,
                        learning_rate=200,
                        init='pca'
                    )
                except TypeError as e2:
                    # If still fails, use simplified parameters
                    print(f"Using simplified TSNE parameters: {e2}")
                    tsne = TSNE(
                        n_components=n_components,
                        perplexity=perplexity,
                        random_state=random_state,
                        init='pca'
                    )
            else:
                # Other errors, re-raise
                raise e
        
        tsne_result = tsne.fit_transform(features_subset)
        
        print(f"t-SNE computation completed:")
        print(f"  Input dimension: {features_subset.shape[1]}")
        print(f"  Output dimension: {tsne_result.shape[1]}")
        print(f"  Sample count: {tsne_result.shape[0]}")
        
        return tsne_result, labels_subset
    
    def visualize_individual_modalities(self, output_dir="tsne_results"):
        """
        Visualize individual modality features
        
        Parameters:
        output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("Visualizing Individual Modality Features")
        print("="*60)
        
        # Check if we have feature data
        if len(self.features) == 0 or len(self.labels) == 0:
            print("Warning: No feature data, run extract_features_from_data() first")
            return
        
        # Define feature types and their display names
        # npy1: Quantum features, npy2: BiLSTM features
        feature_types = ['image', 'npy1', 'npy2']
        feature_names = {
            'image': 'Image Features (ResNet-18)',
            'npy1': 'Quantum Features (LIL_QHN)',  # Corrected
            'npy2': 'Sequence Features (BiLSTM)'    # Corrected
        }
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for i, feature_type in enumerate(feature_types):
            if feature_type in self.features and len(self.features[feature_type]) > 0:
                print(f"\nProcessing {feature_type} features...")
                
                # Compute t-SNE
                tsne_result, labels_subset = self.compute_tsne(
                    self.features[feature_type], 
                    perplexity=30
                )
                
                if tsne_result is not None:
                    # Create DataFrame
                    df = pd.DataFrame({
                        't-SNE 1': tsne_result[:, 0],
                        't-SNE 2': tsne_result[:, 1],
                        'Class': labels_subset
                    })
                    
                    # Map numeric labels to class names
                    if self.class_names and len(self.class_names) > 0:
                        label_to_name = {i: name for i, name in enumerate(self.class_names)}
                        df['Class Name'] = df['Class'].map(label_to_name)
                        hue_col = 'Class Name'
                    else:
                        hue_col = 'Class'
                    
                    # Plot subplot
                    ax = axes[i]
                    scatter = sns.scatterplot(
                        data=df,
                        x='t-SNE 1',
                        y='t-SNE 2',
                        hue=hue_col,
                        palette='husl',
                        alpha=0.8,
                        s=70,  # Larger points
                        edgecolor='w',
                        linewidth=1.0,  # Thicker edge
                        ax=ax
                    )
                    
                    # Set subplot title and labels with larger fonts
                    title = feature_names.get(feature_type, feature_type)
                    ax.set_title(f'{title}\n(Samples: {len(tsne_result)})', fontsize=16, fontweight='bold', pad=15)
                    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, labelpad=10)
                    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, labelpad=10)
                    
                    # Adjust axis limits to fit points
                    x_min, x_max = tsne_result[:, 0].min(), tsne_result[:, 0].max()
                    y_min, y_max = tsne_result[:, 1].min(), tsne_result[:, 1].max()
                    x_margin = (x_max - x_min) * 0.1
                    y_margin = (y_max - y_min) * 0.1
                    ax.set_xlim(x_min - x_margin, x_max + x_margin)
                    ax.set_ylim(y_min - y_margin, y_max + y_margin)
                    
                    # Add grid
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    # Legend handling
                    if i == 0:
                        # Place legend outside the plot
                        ax.legend(title='Classes', fontsize=12, title_fontsize=13, 
                                 bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                    else:
                        ax.legend().remove()
                    
                    # Save individual modality plot
                    save_path = os.path.join(output_dir, f'modality_{feature_type}.png')
                    self.save_single_tsne_plot(
                        tsne_result, 
                        labels_subset,
                        title=f'{title} Visualization from Twitter16',
                        save_path=save_path
                    )
        
        plt.suptitle('Individual Modality Feature Visualizations', fontsize=18, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        # Save comparison plot
        compare_path = os.path.join(output_dir, 'individual_modalities_comparison.png')
        plt.savefig(compare_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.show()
        
        print(f"\nComparison plot saved to: {compare_path}")
    
    def visualize_fused_features(self, output_dir="tsne_results"):
        """
        Visualize fused features
        
        Parameters:
        output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("Visualizing Fused Features")
        print("="*60)
        
        # Check if we have fused features
        if 'fused' not in self.features or len(self.features['fused']) == 0:
            print("Warning: No fused features available")
            return
        
        # Check if we have spherical fused features
        if 'spherical_fused' not in self.features or len(self.features['spherical_fused']) == 0:
            print("Warning: No spherical fused features available")
            return
        
        # Create figure with subplots for fused features
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Fused features visualization
        print("\nProcessing fused features...")
        tsne_result_fused, labels_fused = self.compute_tsne(
            self.features['fused'], 
            perplexity=30
        )
        
        if tsne_result_fused is not None:
            df_fused = pd.DataFrame({
                't-SNE 1': tsne_result_fused[:, 0],
                't-SNE 2': tsne_result_fused[:, 1],
                'Class': labels_fused
            })
            
            if self.class_names and len(self.class_names) > 0:
                label_to_name = {i: name for i, name in enumerate(self.class_names)}
                df_fused['Class Name'] = df_fused['Class'].map(label_to_name)
                hue_col_fused = 'Class Name'
            else:
                hue_col_fused = 'Class'
            
            ax1 = axes[0]
            scatter1 = sns.scatterplot(
                data=df_fused,
                x='t-SNE 1',
                y='t-SNE 2',
                hue=hue_col_fused,
                palette='husl',
                alpha=0.8,
                s=70,
                edgecolor='w',
                linewidth=1.0,
                ax=ax1
            )
            
            ax1.set_title('Model Fused Features\n(Samples: {})'.format(len(tsne_result_fused)), 
                         fontsize=16, fontweight='bold', pad=15)
            ax1.set_xlabel('t-SNE Dimension 1', fontsize=14, labelpad=10)
            ax1.set_ylabel('t-SNE Dimension 2', fontsize=14, labelpad=10)
            
            # Adjust axis limits
            x_min1, x_max1 = tsne_result_fused[:, 0].min(), tsne_result_fused[:, 0].max()
            y_min1, y_max1 = tsne_result_fused[:, 1].min(), tsne_result_fused[:, 1].max()
            x_margin1 = (x_max1 - x_min1) * 0.1
            y_margin1 = (y_max1 - y_min1) * 0.1
            ax1.set_xlim(x_min1 - x_margin1, x_max1 + x_margin1)
            ax1.set_ylim(y_min1 - y_margin1, y_max1 + y_margin1)
            
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(title='Classes', fontsize=12, title_fontsize=13, 
                      bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            
            # Save individual fused features plot
            save_path_fused = os.path.join(output_dir, 'fused_features.png')
            self.save_single_tsne_plot(
                tsne_result_fused, 
                labels_fused,
                title='Model Fused Features Visualization',
                save_path=save_path_fused
            )
        
        # Spherical fused features visualization
        print("\nProcessing spherical fused features...")
        tsne_result_spherical, labels_spherical = self.compute_tsne(
            self.features['spherical_fused'], 
            perplexity=30
        )
        
        if tsne_result_spherical is not None:
            df_spherical = pd.DataFrame({
                't-SNE 1': tsne_result_spherical[:, 0],
                't-SNE 2': tsne_result_spherical[:, 1],
                'Class': labels_spherical
            })
            
            if self.class_names and len(self.class_names) > 0:
                label_to_name = {i: name for i, name in enumerate(self.class_names)}
                df_spherical['Class Name'] = df_spherical['Class'].map(label_to_name)
                hue_col_spherical = 'Class Name'
            else:
                hue_col_spherical = 'Class'
            
            ax2 = axes[1]
            scatter2 = sns.scatterplot(
                data=df_spherical,
                x='t-SNE 1',
                y='t-SNE 2',
                hue=hue_col_spherical,
                palette='husl',
                alpha=0.8,
                s=70,
                edgecolor='w',
                linewidth=1.0,
                ax=ax2
            )
            
            ax2.set_title('VC-BS Fused Features\n(Samples: {})'.format(len(tsne_result_spherical)), 
                         fontsize=16, fontweight='bold', pad=15)
            ax2.set_xlabel('t-SNE Dimension 1', fontsize=14, labelpad=10)
            ax2.set_ylabel('t-SNE Dimension 2', fontsize=14, labelpad=10)
            
            # Adjust axis limits
            x_min2, x_max2 = tsne_result_spherical[:, 0].min(), tsne_result_spherical[:, 0].max()
            y_min2, y_max2 = tsne_result_spherical[:, 1].min(), tsne_result_spherical[:, 1].max()
            x_margin2 = (x_max2 - x_min2) * 0.1
            y_margin2 = (y_max2 - y_min2) * 0.1
            ax2.set_xlim(x_min2 - x_margin2, x_max2 + x_margin2)
            ax2.set_ylim(y_min2 - y_margin2, y_max2 + y_margin2)
            
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(title='Classes', fontsize=12, title_fontsize=13,
                      bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            
            # Save individual spherical fused features plot
            save_path_spherical = os.path.join(output_dir, 'spherical_fused_features.png')
            self.save_single_tsne_plot(
                tsne_result_spherical, 
                labels_spherical,
                title='VC-BS Fused Features Visualization from Twitter16',
                save_path=save_path_spherical
            )
        
        plt.suptitle('Fused Feature Visualizations', fontsize=18, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        # Save comparison plot
        compare_path = os.path.join(output_dir, 'fused_features_comparison.png')
        plt.savefig(compare_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.show()
        
        print(f"\nComparison plot saved to: {compare_path}")
    
    def save_single_tsne_plot(self, tsne_result, labels, title="t-SNE Visualization", save_path=None):
        """
        Save a single t-SNE plot with proper scaling
        
        Parameters:
        tsne_result: t-SNE reduction result
        labels: Labels
        title: Plot title
        save_path: Save path
        """
        if tsne_result is None or len(tsne_result) == 0:
            print("Warning: t-SNE result is empty")
            return
        
        # Create DataFrame
        df = pd.DataFrame({
            't-SNE 1': tsne_result[:, 0],
            't-SNE 2': tsne_result[:, 1],
            'Class': labels
        })
        
        # Map numeric labels to class names
        if self.class_names and len(self.class_names) > 0:
            label_to_name = {i: name for i, name in enumerate(self.class_names)}
            df['Class Name'] = df['Class'].map(label_to_name)
            hue_col = 'Class Name'
        else:
            hue_col = 'Class'
        
        # Create figure with larger size
        plt.figure(figsize=(12, 10))
        
        # Plot scatter with larger points
        scatter = sns.scatterplot(
            data=df,
            x='t-SNE 1',
            y='t-SNE 2',
            hue=hue_col,
            palette='husl',
            alpha=0.8,
            s=80,  # Larger points
            edgecolor='w',
            linewidth=1.5  # Thicker edge
        )
        
        # Set title and labels with larger fonts
        plt.title(title, fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('t-SNE Dimension 1', fontsize=16, labelpad=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=16, labelpad=12)
        
        # Adjust axis limits to fit points with margin
        x_min, x_max = tsne_result[:, 0].min(), tsne_result[:, 0].max()
        y_min, y_max = tsne_result[:, 1].min(), tsne_result[:, 1].max()
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend with larger font, placed outside
        plt.legend(title='Classes', fontsize=13, title_fontsize=14,
                  bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def visualize_class_separation(self, output_dir="tsne_results"):
        """
        Visualize class separation for each class
        
        Parameters:
        output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("Visualizing Class Separation")
        print("="*60)
        
        # Check if we have spherical fused features
        if 'spherical_fused' not in self.features or len(self.features['spherical_fused']) == 0:
            print("Warning: No spherical fused features")
            return
        
        fused_features = self.features['spherical_fused']
        
        # Compute t-SNE
        tsne_result, labels_subset = self.compute_tsne(fused_features, perplexity=30)
        
        if tsne_result is None:
            return
        
        # Create separate plot for each class
        for class_idx, class_name in enumerate(self.class_names):
            if class_idx >= len(self.class_names):
                continue
                
            # Create mask
            class_mask = labels_subset == class_idx
            other_mask = ~class_mask
            
            if np.sum(class_mask) == 0:
                print(f"Warning: Class {class_name} has no samples")
                continue
            
            plt.figure(figsize=(12, 10))
            
            # Plot other classes
            if np.sum(other_mask) > 0:
                plt.scatter(
                    tsne_result[other_mask, 0],
                    tsne_result[other_mask, 1],
                    c='lightgray',
                    alpha=0.3,
                    s=50,
                    label='Other Classes'
                )
            
            # Plot current class
            plt.scatter(
                tsne_result[class_mask, 0],
                tsne_result[class_mask, 1],
                c='red',
                alpha=0.8,
                s=100,
                edgecolors='darkred',
                linewidth=2.0,
                label=class_name
            )
            
            # Set title and labels with larger fonts
            plt.title(f'Class "{class_name}" Feature Distribution\n(Samples: {np.sum(class_mask)})', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('t-SNE Dimension 1', fontsize=14, labelpad=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=14, labelpad=12)
            
            # Adjust axis limits
            x_min, x_max = tsne_result[:, 0].min(), tsne_result[:, 0].max()
            y_min, y_max = tsne_result[:, 1].min(), tsne_result[:, 1].max()
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            plt.xlim(x_min - x_margin, x_max + x_margin)
            plt.ylim(y_min - y_margin, y_max + y_margin)
            
            plt.legend(fontsize=13, loc='upper left')
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # Save plot
            save_path = os.path.join(output_dir, f'class_{class_name}_separation.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
            plt.show()
            
            print(f"Class {class_name} separation plot saved to: {save_path}")
    
    def save_features(self, output_path="extracted_features.npz"):
        """
        Save extracted features
        
        Parameters:
        output_path: Output file path
        """
        if len(self.features) == 0 or len(self.labels) == 0:
            print("Warning: No feature data to save")
            return
        
        # Prepare save data
        save_data = {
            'labels': self.labels,
            'class_names': np.array(self.class_names, dtype=object),
            'sample_info': self.sample_info,
            'model_config': self.model_config
        }
        
        # Add features
        for key, value in self.features.items():
            if len(value) > 0:
                save_data[key] = value
        
        # Save as npz file
        np.savez_compressed(output_path, **save_data)
        
        print(f"\nFeatures saved to: {output_path}")
        print(f"Saved feature types: {list(self.features.keys())}")
        print(f"Total samples: {len(self.labels)}")
    
    def load_features(self, input_path="extracted_features.npz"):
        """
        Load features from file
        
        Parameters:
        input_path: Input file path
        """
        if not os.path.exists(input_path):
            print(f"Warning: Feature file does not exist: {input_path}")
            return False
        
        # Load data
        data = np.load(input_path, allow_pickle=True)
        
        # Extract features
        self.features = {}
        for key in data.files:
            if key not in ['labels', 'class_names', 'sample_info', 'model_config']:
                self.features[key] = data[key]
        
        # Extract labels and other information
        self.labels = data['labels']
        self.class_names = data['class_names'].tolist()
        self.sample_info = data['sample_info'].tolist()
        self.model_config = data['model_config'].item() if 'model_config' in data else {}
        
        print(f"\nFeatures loaded from {input_path}")
        print(f"Loaded feature types: {list(self.features.keys())}")
        print(f"Total samples: {len(self.labels)}")
        print(f"Classes: {self.class_names}")
        
        return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='T-SNE Visualization Tool')
    parser.add_argument('--model-path', type=str, required=True,
                       help='T_resnet model path')
    parser.add_argument('--config-path', type=str, 
                       help='Configuration file path')
    parser.add_argument('--image-dir', type=str,
                       help='Image directory')
    parser.add_argument('--npy1-dir', type=str,
                       help='Quantum feature directory (npy1)')
    parser.add_argument('--npy2-dir', type=str,
                       help='BiLSTM feature directory (npy2)')
    parser.add_argument('--features-file', type=str,
                       help='Pre-extracted feature file path')
    parser.add_argument('--output-dir', type=str, default='tsne_results',
                       help='Output directory')
    parser.add_argument('--max-samples', type=int, default=200,
                       help='Maximum samples per class')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip feature extraction, use feature file directly')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = TSNEVisualizer(args.model_path, args.config_path)
    
    # Extract or load features
    if args.skip_extraction and args.features_file:
        print("Skipping feature extraction, loading features from file...")
        visualizer.load_features(args.features_file)
    elif args.image_dir and args.npy1_dir and args.npy2_dir:
        print("Extracting features from dataset...")
        visualizer.extract_features_from_data(
            args.image_dir, 
            args.npy1_dir,  # Quantum features
            args.npy2_dir,  # BiLSTM features
            max_samples_per_class=args.max_samples
        )
        
        # Save extracted features
        features_file = os.path.join(args.output_dir, 'extracted_features.npz')
        visualizer.save_features(features_file)
    else:
        print("Error: Need to provide feature file or dataset paths")
        return
    
    # Visualize individual modality features
    visualizer.visualize_individual_modalities(args.output_dir)
    
    # Visualize fused features
    visualizer.visualize_fused_features(args.output_dir)
    
    # Visualize class separation
    visualizer.visualize_class_separation(args.output_dir)
    
    print("\n" + "="*60)
    print("T-SNE Visualization Complete!")
    print(f"All results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()