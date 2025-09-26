import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
import json
import pickle
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import current_app  # ← Add this import
from sklearn.ensemble import RandomForestClassifier  # Add this import
import os  # ← Add this import

class PrepaymentDataUploader:
    """
    Handles data upload for prepayment models
    """
    def __init__(self, upload_folder='derivapro/static/uploads'):
        self.upload_folder = upload_folder
        self.allowed_extensions = {'csv'}
        
        # Ensure upload directory exists
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def allowed_file(self, filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def save_uploaded_file(self, file):
        if not file or file.filename == '':
            return {'success': False, 'error': 'No file selected'}
        if not self.allowed_file(file.filename):
            return {'success': False, 'error': 'Invalid file type. Please upload CSV files only.'}
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(self.upload_folder, filename)
            file.save(filepath)
            return {'success': True, 'filepath': filepath, 'filename': filename, 'error': None}
        except Exception as e:
            return {'success': False, 'error': f'Error saving file: {str(e)}'}    
    
    def delete_file(self, filepath):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                # Also delete metadata file if it exists
                metadata_file = filepath.replace('.csv', '_metadata.json')
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                return {'success': True, 'error': None}
            else:
                return {'success': False, 'error': 'File not found'}
        except Exception as e:
            return {'success': False, 'error': f'Error deleting file: {str(e)}'}

class Validation:
    """
    Provides summaries and plots similar to ipywidgets workflow, 
    but for Flask HTML templates.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        self.metadata_file = filepath.replace('.csv', '_metadata.json')
        self._load_conversions()
        
        # Apply conversions first, then imputations to handle any NaN values created by conversions
        self._apply_stored_conversions()
        self._apply_stored_imputations()
        
        # Recreate splits after all data processing is complete
        if hasattr(self, 'target') and self.target and self.target in self.data.columns:
            self._recreate_data_split()

    def _load_conversions(self):
        """Load stored conversion metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self._conversions = metadata.get('conversions', {})
                    self._imputations = metadata.get('imputations', {})
                    self._normalizations = metadata.get('normalizations', {})
                    self._data_split_info = metadata.get('data_split', {})
                    self._selected_modeling_columns = metadata.get('selected_modeling_columns', [])  # Add this line
                    
                    # Restore split attributes if they exist
                    if self._data_split_info:
                        self.target = self._data_split_info.get('target')
                        self.task = self._data_split_info.get('task')
                        self.test_ratio = self._data_split_info.get('test_ratio')
                        self.random = self._data_split_info.get('random_state')
            except:
                self._conversions = {}
                self._imputations = {}
                self._normalizations = {}
                self._data_split_info = {}
        else:
            self._conversions = {}
            self._imputations = {}
            self._normalizations = {}
            self._data_split_info = {}
    
    def _save_conversions(self):
        """Save conversion and imputation metadata"""
        metadata = {
            'conversions': self._conversions,
            'imputations': self._imputations,
            'normalizations': self._normalizations,
            'data_split': getattr(self, '_data_split_info', {}),
            'selected_modeling_columns': getattr(self, '_selected_modeling_columns', []),
            'registered_model_info': getattr(self, '_registered_model_info', {})  # Add this line
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)
    
    def _recreate_data_split(self):
        """Recreate the data split using stored parameters"""
        from sklearn.model_selection import train_test_split
        
        try:
            # Prepare features and target
            X = self.data.drop(columns=[self.target])
            y = self.data[self.target]
            
            # Recreate the split with same parameters
            if self.task == 'classification':
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=self.test_ratio, random_state=self.random, stratify=y
                )
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=self.test_ratio, random_state=self.random
                )
            
            # Apply column selection if it was previously applied
            if hasattr(self, '_selected_modeling_columns') and self._selected_modeling_columns:
                # Validate that all selected columns exist in the training data
                available_cols = [col for col in self._selected_modeling_columns if col in self.X_train.columns]
                if available_cols:
                    self.X_train = self.X_train[available_cols]
                    self.X_test = self.X_test[available_cols]
                    print(f"✓ Restored column selection: {len(available_cols)} columns")
                else:
                    print("⚠️  No selected columns found in recreated data split")
                    
        except Exception as e:
            print(f"Error recreating data split: {e}")
            # Reset split attributes if recreation fails
            self.target = None
            self.task = None
            self.test_ratio = None
            self.random = None


    def _apply_stored_conversions(self):
        """Reapply stored conversions after loading CSV"""
        for col, conversion_type in self._conversions.items():
            if col in self.data.columns:
                if conversion_type == 'numeric':
                    if not pd.api.types.is_numeric_dtype(self.data[col]):
                        print(f"Debug: Converting column '{col}' to numeric...")
                        print(f"  Current dtype: {self.data[col].dtype}")
                        print(f"  Sample values: {self.data[col].head().tolist()}")
                        
                        # Simply apply the conversion - let the user decide what's appropriate
                        self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                        print(f"  ✓ Conversion applied")
                    else:
                        print(f"Debug: Column '{col}' is already numeric, skipping conversion")
                elif conversion_type == 'categorical':
                    self.data[col] = self.data[col].astype('category')
                    print(f"Debug: Converted column '{col}' to categorical")
    def _apply_stored_imputations(self):
        """Reapply stored imputations after loading CSV"""
        for col, imputation_info in self._imputations.items():
            if col in self.data.columns:
                method = imputation_info['method']
                value = imputation_info.get('value')
                
                if method == 'mean':
                    if value is not None:
                        self.data[col] = self.data[col].fillna(value)
                    else:
                        self.data[col] = self.data[col].fillna(self.data[col].mean())
                elif method == 'median':
                    if value is not None:
                        self.data[col] = self.data[col].fillna(value)
                    else:
                        self.data[col] = self.data[col].fillna(self.data[col].median())
                elif method == 'mode':
                    if value is not None:
                        self.data[col] = self.data[col].fillna(value)
                    else:
                        mode_values = self.data[col].mode()
                        if len(mode_values) > 0:
                            self.data[col] = self.data[col].fillna(mode_values[0])
                elif method == 'constant':
                    if value is not None:
                        self.data[col] = self.data[col].fillna(value)

    def summary_numerical(self):
        num_vars = self.data.select_dtypes(include=[np.number])
        if num_vars.empty:
            return "<p>No numerical features found.</p>"
        return num_vars.describe().transpose().to_html(classes="table table-striped")

    def summary_categorical(self):
        cat_vars = self.data.select_dtypes(include=['object', 'category', 'bool'])
        if cat_vars.empty:
            return "<p>No categorical features found.</p>"
        return cat_vars.describe().transpose().to_html(classes="table table-striped")

    def missing_values(self):
        nulls = self.data.isnull().any()
        return nulls.to_frame("Has Missing Values").to_html(classes="table table-striped")

    def plot_distribution(self, column):
        fig, ax = plt.subplots(figsize=(6,4))
        if self.data[column].dtype == 'object' or self.data[column].dtype.name == 'category':
            sns.countplot(data=self.data, x=column, ax=ax)
        else:
            sns.histplot(data=self.data, x=column, ax=ax)
        plt.title(f"Distribution of {column}")
        return self._fig_to_base64(fig)

    def plot_scatter(self, col1, col2):
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(data=self.data, x=col1, y=col2, ax=ax)
        plt.title(f"{col1} vs {col2}")
        return self._fig_to_base64(fig)

    def plot_heatmap(self):
        corr = self.data.select_dtypes(include=[np.number]).corr()
        
        # Dynamic sizing based on number of columns
        n_cols = len(corr.columns)
        fig_size = max(8, n_cols * 0.8)  # Minimum 8, scale with number of columns
        
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(corr, annot=True, cmap="coolwarm", cbar=True, ax=ax, 
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig):
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return encoded

    def convert_columns(self, columns, conversion_type):
        """Convert specified columns to the given type"""
        for col in columns:
            if conversion_type == 'numeric':
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self._conversions[col] = 'numeric'
            elif conversion_type == 'categorical':
                self.data[col] = self.data[col].astype('category')
                self._conversions[col] = 'categorical'
        
        # Save metadata
        self._save_conversions()
    
    def impute_data(self, columns, imputation_method):
        for col in columns:
            # Store the imputation value for later reapplication
            imputation_value = None
            
            if imputation_method == 'mean':
                imputation_value = self.data[col].mean()
                self.data[col] = self.data[col].fillna(imputation_value)
            elif imputation_method == 'median':
                imputation_value = self.data[col].median()
                self.data[col] = self.data[col].fillna(imputation_value)
            elif imputation_method == 'mode':
                mode_values = self.data[col].mode()
                if len(mode_values) > 0:
                    imputation_value = mode_values[0]
                    self.data[col] = self.data[col].fillna(imputation_value)
                else:
                    # Fallback to most frequent value or first non-null value
                    if not self.data[col].dropna().empty:
                        imputation_value = self.data[col].dropna().iloc[0]
                        self.data[col] = self.data[col].fillna(imputation_value)
            elif imputation_method == 'constant':
                first_non_null = self.data[col].dropna()
                if not first_non_null.empty:
                    imputation_value = first_non_null.iloc[0]
                    self.data[col] = self.data[col].fillna(imputation_value)
            
            # Store imputation metadata
            if imputation_value is not None:
                self._imputations[col] = {
                    'method': imputation_method,
                    'value': imputation_value
                }
        
        # Save metadata
        self._save_conversions()
        return self.data

    def normalize_data(self, columns, normalization_method):
        """Normalize specified columns using the given method"""
        for col in columns:
            if col not in self.data.columns:
                continue
                
            if normalization_method == 'standard':
                # Z-score normalization: (x - mean) / std
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                if std_val != 0:  # Avoid division by zero
                    self.data[col] = (self.data[col] - mean_val) / std_val
                
                # Store parameters
                self._normalizations[col] = {
                    'method': 'standard',
                    'mean': float(mean_val),
                    'std': float(std_val)
                }
                
            elif normalization_method == 'minmax':
                # Min-max normalization: (x - min) / (max - min)
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                range_val = max_val - min_val
                if range_val != 0:  # Avoid division by zero
                    self.data[col] = (self.data[col] - min_val) / range_val
                
                # Store parameters
                self._normalizations[col] = {
                    'method': 'minmax',
                    'min': float(min_val),
                    'max': float(max_val)
                }
        
        # Save metadata
        self._save_conversions()
        return self.data

    def train_test_split_data(self, target_variable, task, train_test_split_ratio, random_state):
        """Split data into training and testing sets"""
        from sklearn.model_selection import train_test_split
        
        if target_variable == 'make selection' or task == 'make selection':
            print("Please make a selection.")
            return False
        
        if target_variable not in self.data.columns:
            print(f"Target variable '{target_variable}' not found in data")
            return False
            
        # Store parameters
        self.target = target_variable
        self.task = task
        self.test_ratio = train_test_split_ratio
        self.random = random_state
        
        # Store split info for persistence
        self._data_split_info = {
            'target': target_variable,
            'task': task,
            'test_ratio': train_test_split_ratio,
            'random_state': random_state
        }
        
        # Prepare features - use selected columns if available
        if hasattr(self, '_selected_modeling_columns') and self._selected_modeling_columns:
            # Use only selected columns for modeling
            available_cols = [col for col in self._selected_modeling_columns if col in self.data.columns and col != target_variable]
            X = self.data[available_cols]
            print(f"Using {len(available_cols)} selected columns for modeling: {available_cols}")
        else:
            # Use all columns except target (fallback to original behavior)
            X = self.data.drop(columns=[target_variable])
            print(f"No column selection applied. Using all {len(X.columns)} available columns.")
        
        y = self.data[target_variable]
        
        # Perform the split
        if task == 'classification':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=train_test_split_ratio, random_state=random_state, stratify=y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=train_test_split_ratio, random_state=random_state
            )
        
        # Save metadata including split info
        self._save_conversions()
        
        return True
    
    def apply_column_selection(self, selected_columns):
        """Apply column selection to existing train/test splits"""
        
        if not hasattr(self, 'X_train') or self.X_train is None:
            raise ValueError("Data must be split first using Data Preparation")
        
        # Filter out the target variable from selected columns if it exists
        modeling_columns = [col for col in selected_columns if col != getattr(self, 'target', None)]
        
        # Validate that all selected columns exist in the training data
        missing_cols = [col for col in modeling_columns if col not in self.X_train.columns]
        if missing_cols:
            raise ValueError(f"Selected columns not found in training data: {missing_cols}")
        
        # Apply column selection to both train and test sets
        self.X_train = self.X_train[modeling_columns]
        self.X_test = self.X_test[modeling_columns]
        
        # Store the selected columns (including target for reference)
        self._selected_modeling_columns = selected_columns
        self._save_conversions()
        
        print(f"Column selection applied. Using {len(modeling_columns)} columns: {modeling_columns}")
        print(f"New X_train shape: {self.X_train.shape}")
        print(f"New X_test shape: {self.X_test.shape}")
        
        return True


    def feature_selection(self, feature_selection_method='pearson', threshold=0.5):
        """
        Perform feature selection based on the method chosen
        Methods:
        - Pearson Correlation
        - Spearman Correlation
        - Feature Importance
        
        """
        try:
            print(f"Starting feature selection with method: {feature_selection_method}, threshold: {threshold}")
            print(f"Target variable: {self.target}")
            
            # CHECK: Use X_train (which respects column selection) instead of self.data
            if hasattr(self, 'X_train') and self.X_train is not None:
                print(f"Using X_train data shape: {self.X_train.shape}")
                feature_data = self.X_train
                target_data = self.y_train
            else:
                print(f"Using full data shape: {self.data.shape}")
                feature_data = self.data.drop(columns=[self.target])
                target_data = self.data[self.target]
                
            print(f"Feature data columns: {list(feature_data.columns)}")
            
            if feature_selection_method == 'pearson':
                # Pearson Correlation - correlate with target, not between features
                num_vars = feature_data.select_dtypes(include=[np.number]).columns
                print(f"Found {len(num_vars)} numerical variables: {list(num_vars)}")
                
                # Filter out columns with zero variance (constant values) and missing values
                valid_vars = []
                for col in num_vars:
                    if (feature_data[col].nunique() > 1 and 
                        not feature_data[col].isna().all() and 
                        feature_data[col].std() > 1e-10):  # Check for near-zero std
                        valid_vars.append(col)
                
                print(f"Valid variables for correlation: {valid_vars}")
                
                if not valid_vars:
                    print("No valid variables found for correlation analysis")
                    return {}, None
                    
                # Calculate correlations only for valid variables
                correlations = {}
                for col in valid_vars:
                    try:
                        corr_val = feature_data[col].corr(target_data)
                        if not pd.isna(corr_val):
                            correlations[col] = corr_val
                            print(f"Correlation for {col}: {corr_val}")
                    except Exception as e:
                        print(f"Error calculating correlation for {col}: {e}")
                        continue
                
                print(f"Final correlations: {correlations}")
                plot = self.create_corr_plot(threshold, valid_vars, feature_data, target_data)
                print(f"Plot generated successfully: {plot is not None}")
                return correlations, plot
                
            elif feature_selection_method == 'spearman':
                # Spearman Correlation
                num_vars = feature_data.select_dtypes(include=[np.number]).columns
                print(f"Found {len(num_vars)} numerical variables: {list(num_vars)}")
                
                # Filter out columns with zero variance and missing values
                valid_vars = []
                for col in num_vars:
                    if (feature_data[col].nunique() > 1 and 
                        not feature_data[col].isna().all() and 
                        feature_data[col].std() > 1e-10):
                        valid_vars.append(col)
                
                if not valid_vars:
                    print("No valid variables found for correlation analysis")
                    return {}, None
                    
                # Calculate correlations only for valid variables
                correlations = {}
                for col in valid_vars:
                    try:
                        corr_val = feature_data[col].corr(target_data, method='spearman')
                        if not pd.isna(corr_val):
                            correlations[col] = corr_val
                    except:
                        # Skip columns that cause correlation errors
                        continue
                        
                return correlations, self.create_spearman_plot(threshold, valid_vars, feature_data, target_data)
            
            elif feature_selection_method == 'feature_importance':
                # Feature Importance - handle both classification and regression
                if getattr(self, 'task', None) == 'classification':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:  # For regression tasks (like predicting Interest Rate)
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                model.fit(self.X_train, self.y_train)
                feature_importance = dict(zip(self.X_train.columns, model.feature_importances_))
                print(f"Feature importance: {feature_importance}")
                return feature_importance, self.create_importance_plot(threshold)
                
        except Exception as e:
            print(f"Error in feature_selection: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def create_corr_plot(self, threshold, valid_vars=None, feature_data=None, target_data=None):
        """Create Pearson correlation plot between features and target"""
        try:
            # Select only numerical variables for pearson correlation
            if valid_vars is None:
                # Use provided data or fallback to self.data
                if feature_data is not None:
                    num_vars = feature_data.select_dtypes(include=[np.number]).columns
                    data_to_use = feature_data
                else:
                    num_vars = self.data.drop(columns=[self.target]).select_dtypes(include=[np.number]).columns
                    data_to_use = self.data
                
                # Filter out problematic columns
                valid_vars = []
                for col in num_vars:
                    if (data_to_use[col].nunique() > 1 and 
                        not data_to_use[col].isna().all() and 
                        data_to_use[col].std() > 1e-10):
                        valid_vars.append(col)
            
            if not valid_vars:
                # Create empty plot if no valid variables
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.5, 'No valid numerical variables for correlation analysis', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'Correlation Analysis with Target Variable: {self.target}', fontsize=14)
                return self._fig_to_base64(fig)
            
            # Calculate correlations safely
            correlations = {}
            for col in valid_vars:
                try:
                    # Use provided data if available, otherwise use self.data
                    if feature_data is not None and target_data is not None:
                        corr_val = feature_data[col].corr(target_data)
                    else:
                        corr_val = self.data[col].corr(self.data[self.target])
                    if not pd.isna(corr_val):
                        correlations[col] = corr_val
                except Exception as e:
                    print(f"Error calculating correlation for {col}: {e}")
                    continue
            
            if not correlations:
                # Create empty plot if no correlations could be calculated
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.5, 'Unable to calculate correlations', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'Correlation Analysis with Target Variable: {self.target}', fontsize=14)
                return self._fig_to_base64(fig)
            
            # Convert to Series for easier manipulation
            corr = pd.Series(correlations)
            
            # Filter correlations by threshold
            corr_filtered = corr[abs(corr) >= threshold]
            corr_filtered_less = corr[abs(corr) < threshold]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create the bar plot
            if len(corr_filtered_less) > 0:
                bars_below = ax.bar(x=corr_filtered_less.index, height=corr_filtered_less.values, 
                                color='lightgray', alpha=0.6, label=f'|correlation| < {threshold}')
            if len(corr_filtered) > 0:
                bars_above = ax.bar(x=corr_filtered.index, height=corr_filtered.values, 
                                color='green', alpha=0.8, label=f'|correlation| ≥ {threshold}')
            
            ax.tick_params(axis='x', rotation=45)
            ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold: ±{threshold}')
            ax.axhline(y=-threshold, color='r', linestyle='--', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            ax.set_title(f'Pearson Correlation with Target Variable: {self.target}', fontsize=14)
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Correlation Coefficient', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # return self._save_plot_to_static(fig, 'feature_selection')
            # To this:
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"Error creating correlation plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_spearman_plot(self, threshold, valid_vars=None, feature_data=None, target_data=None):
        """Create Spearman correlation plot between features and target"""
        try:
            # Select only numerical variables for spearman correlation
            if valid_vars is None:
                # Use provided data or fallback to self.data
                if feature_data is not None:
                    num_vars = feature_data.select_dtypes(include=[np.number]).columns
                    data_to_use = feature_data
                else:
                    num_vars = self.data.drop(columns=[self.target]).select_dtypes(include=[np.number]).columns
                    data_to_use = self.data
                
                # Filter out problematic columns
                valid_vars = []
                for col in num_vars:
                    if (data_to_use[col].nunique() > 1 and 
                        not data_to_use[col].isna().all() and 
                        data_to_use[col].std() > 1e-10):
                        valid_vars.append(col)
            
            if not valid_vars:
                # Create empty plot if no valid variables
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.5, 'No valid numerical variables for Spearman correlation analysis', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'Spearman Correlation Analysis with Target Variable: {self.target}', fontsize=14)
                return self._fig_to_base64(fig)
            
            # Calculate Spearman correlations safely
            correlations = {}
            for col in valid_vars:
                try:
                    # Use provided data if available, otherwise use self.data
                    if feature_data is not None and target_data is not None:
                        corr_val = feature_data[col].corr(target_data, method='spearman')
                    else:
                        corr_val = self.data[col].corr(self.data[self.target], method='spearman')
                    if not pd.isna(corr_val):
                        correlations[col] = corr_val
                except Exception as e:
                    print(f"Error calculating Spearman correlation for {col}: {e}")
                    continue
            
            if not correlations:
                # Create empty plot if no correlations could be calculated
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.5, 'Unable to calculate Spearman correlations', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'Spearman Correlation Analysis with Target Variable: {self.target}', fontsize=14)
                return self._fig_to_base64(fig)
            
            # Convert to Series for easier manipulation
            corr = pd.Series(correlations)
            
            # Filter correlations by threshold
            corr_filtered = corr[abs(corr) >= threshold]
            corr_filtered_less = corr[abs(corr) < threshold]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create the bar plot with different color scheme for Spearman
            if len(corr_filtered_less) > 0:
                bars_below = ax.bar(x=corr_filtered_less.index, height=corr_filtered_less.values, 
                                color='lightcoral', alpha=0.6, label=f'|correlation| < {threshold}')
            if len(corr_filtered) > 0:
                bars_above = ax.bar(x=corr_filtered.index, height=corr_filtered.values, 
                                color='darkblue', alpha=0.8, label=f'|correlation| ≥ {threshold}')
            
            ax.tick_params(axis='x', rotation=45)
            ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold: ±{threshold}')
            ax.axhline(y=-threshold, color='r', linestyle='--', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            ax.set_title(f'Spearman Correlation with Target Variable: {self.target}', fontsize=14)
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Spearman Correlation Coefficient', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add subtitle explaining Spearman correlation
            fig.suptitle('Spearman Rank Correlation (measures monotonic relationships)', 
                        fontsize=10, y=0.02, alpha=0.7)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # return self._save_plot_to_static(fig, 'spearman_correlation')
            return self._fig_to_base64(fig)  # ← Change this back to _fig_to_base64
            
        except Exception as e:
            print(f"Error creating Spearman correlation plot: {e}")
            import traceback
            traceback.print_exc()
            return None


    def create_importance_plot(self, threshold):
        """Create feature importance plot"""
        try:
            if not hasattr(self, 'X_train') or self.X_train is None:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.5, 'Data must be split before feature importance analysis', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Feature Importance Analysis Error', fontsize=14)
                return self._fig_to_base64(fig)
            
            # Use the same logic as in feature_selection method
            if getattr(self, 'task', None) == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # For regression tasks
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(self.X_train, self.y_train)
            
            feature_importance = dict(zip(self.X_train.columns, model.feature_importances_))
            
            # Convert to Series for easier manipulation
            importance = pd.Series(feature_importance)
            
            # Filter importance by threshold
            importance_filtered = importance[importance >= threshold]
            importance_filtered_less = importance[importance < threshold]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create the bar plot
            if len(importance_filtered_less) > 0:
                bars_below = ax.bar(x=importance_filtered_less.index, height=importance_filtered_less.values, 
                                color='lightgray', alpha=0.6, label=f'Importance < {threshold}')
            if len(importance_filtered) > 0:
                bars_above = ax.bar(x=importance_filtered.index, height=importance_filtered.values, 
                                color='blue', alpha=0.8, label=f'Importance ≥ {threshold}')
            
            ax.tick_params(axis='x', rotation=45)
            ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold: {threshold}')
            
            ax.set_title(f'Feature Importance Analysis', fontsize=14)
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Importance Score', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Adjust layout to prevent label cutoff
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            fig.tight_layout()
            
            # return self._save_plot_to_static(fig, 'feature_importance')
            return self._fig_to_base64(fig)  # ← Change this back to _fig_to_base64
        except Exception as e:
            print(f"Error in create_importance_plot: {e}")
            # Return a simple error plot
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f'Error creating importance plot: {str(e)}', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Feature Importance Analysis Error', fontsize=14)
            # return self._save_plot_to_static(fig, 'importance_error')
            return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig):
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return encoded

    def _save_plot_to_static(self, fig, plot_type='feature_selection'):
        """Save plot to static folder and return the file path"""
        import uuid
        import os
        
        # Create static/plots directory if it doesn't exist
        # plots_dir = 'derivapro/static/plots'
        # os.makedirs(plots_dir, exist_ok=True)
        # Create static/plots directory if it doesn't exist - use app-relative path
        plots_dir = os.path.join(current_app.static_folder, 'plots')  # ← Change this line
        os.makedirs(plots_dir, exist_ok=True)
    
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]  # Short unique ID
        filename = f"{plot_type}_{unique_id}.png"
        filepath = os.path.join(plots_dir, filename)
        
        # Save the plot
        fig.tight_layout()
        fig.savefig(filepath, format="png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Return the web-accessible path
        return f"plots/{filename}"
    

    # final column selection from the list of features after feature selection
    def final_column_selection(self, selected_columns):
        """Apply column selection to existing train/test splits"""
        
        if not hasattr(self, 'X_train') or self.X_train is None:
            raise ValueError("Data must be split first using Data Preparation")
        
        # Filter out the target variable from selected columns if it exists
        modeling_columns = [col for col in selected_columns if col != getattr(self, 'target', None)]
        
        # Validate that all selected columns exist in the training data
        missing_cols = [col for col in modeling_columns if col not in self.X_train.columns]
        if missing_cols:
            raise ValueError(f"Selected columns not found in training data: {missing_cols}")
        
        # Apply column selection to both train and test sets
        self.X_train = self.X_train[modeling_columns]
        self.X_test = self.X_test[modeling_columns]
        
        # Store the selected columns (including target for reference)
        self._selected_modeling_columns = selected_columns
        self._save_conversions()
        
        print(f"Final column selection applied. Using {len(modeling_columns)} columns: {modeling_columns}")
        print(f"New X_train shape: {self.X_train.shape}")
        print(f"New X_test shape: {self.X_test.shape}")
        
        return True


    def model_training(self, model_training_method, hyperparameters=None):
        # Check if data has been properly split first
        if not hasattr(self, 'X_train') or self.X_train is None:
            raise ValueError("Data must be split first using the Data Preparation step")
        
        # Add debugging information
        print(f"Original data shape: {self.data.shape}")
        print(f"X_train shape before cleaning: {self.X_train.shape}")
        print(f"y_train shape before cleaning: {self.y_train.shape}")
        
        # CHECK: Is column selection applied?
        if hasattr(self, '_selected_modeling_columns'):
            print(f"✓ Column selection applied: {len(self._selected_modeling_columns)} columns selected")
            print(f"Selected columns: {self._selected_modeling_columns}")
            print(f"X_train columns: {list(self.X_train.columns)}")
            
            # Verify that X_train only contains selected columns
            if set(self.X_train.columns) == set(self._selected_modeling_columns):
                print("✓ X_train matches selected columns")
            else:
                print("⚠️  X_train does NOT match selected columns!")
                print(f"  Missing: {set(self._selected_modeling_columns) - set(self.X_train.columns)}")
                print(f"  Extra: {set(self.X_train.columns) - set(self._selected_modeling_columns)}")
        else:
            print("⚠️  No column selection applied - using all available columns")
        
        # Use the existing train/test splits from data preparation (already imputed)
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        y_train = self.y_train.copy()
        y_test = self.y_test.copy()
        task = self.task
        
        # Check for any remaining NaN values and report them
        print(f"Checking for NaN values in training data...")
        print(f"X_train NaN count: {X_train.isnull().sum().sum()}")
        print(f"y_train NaN count: {y_train.isnull().sum()}")
        
        if X_train.isnull().sum().sum() > 0 or y_train.isnull().sum() > 0:
            print("Warning: Found NaN values in training data. Dropping rows with NaN values...")
            # Create a combined dataset to drop NaN rows consistently
            import pandas as pd
            combined = pd.concat([X_train, y_train], axis=1)
            print(f"Combined dataset shape before cleaning: {combined.shape}")
            combined_clean = combined.dropna()
            print(f"Combined dataset shape after cleaning: {combined_clean.shape}")
            
            # Check if we have any data left
            if combined_clean.empty:
                raise ValueError("All data was removed during NaN cleaning. Check your data quality or imputation strategy.")
            
            X_train = combined_clean.drop(columns=[y_train.name])
            y_train = combined_clean[y_train.name]
            print(f"After cleaning - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # Set default hyperparameters or use provided ones
        if hyperparameters is None:
            hyperparameters = {}
        
        # Default parameters for each model
        # Update the default parameters to include new ones
        default_params = {
            'Logistic Regression': {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 1000
            },
            'Random Forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'GBM': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'subsample': 1.0,
                'random_state': 42
            },
            'LightGBM': {
                'boosting_type': 'gbdt',
                'n_estimators': 100,
                'num_leaves': 31,
                'max_depth': -1,
                'learning_rate': 0.1,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'random_state': 42
            }
        }
        
        # Get model name from method parameter
        model_name_mapping = {
            'logistic_regression': 'Logistic Regression',
            'random_forest': 'Random Forest', 
            'gradient_boosting': 'GBM',
            'lightgbm': 'LightGBM'
        }
        
        model_name = model_name_mapping.get(model_training_method)
        if not model_name:
            raise ValueError(f"Unknown model training method: {model_training_method}")
        
        # Merge default parameters with user-provided hyperparameters
        params = default_params[model_name].copy()
        params.update(hyperparameters)

        if task == 'classification':
            # Define models for classification task
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            try:
                import lightgbm as lgb
            except ImportError:
                lgb = None
                
            if model_name == 'Logistic Regression':
                model = LogisticRegression(**params)
            elif model_name == 'Random Forest':
                model = RandomForestClassifier(**params)
            elif model_name == 'GBM':
                model = GradientBoostingClassifier(**params)
            elif model_name == 'LightGBM':
                if lgb is None:
                    raise ImportError("LightGBM is not installed")
                model = lgb.LGBMClassifier(**params)
            
            # Define metrics for classification task
            from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
            metrics = {
                'AUC': roc_auc_score,
                'Accuracy': accuracy_score,
                'Precision': precision_score,
                'Recall': recall_score,
                'F1 Score': f1_score,
            }
        elif task == "regression":
            # Define models for regression task
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            try:
                import lightgbm as lgb
            except ImportError:
                lgb = None
                
            if model_name == 'Logistic Regression':
                # For regression, use Linear Regression instead
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            elif model_name == 'Random Forest':
                model = RandomForestRegressor(**params)
            elif model_name == 'GBM':
                model = GradientBoostingRegressor(**params)
            elif model_name == 'LightGBM':
                if lgb is None:
                    raise ImportError("LightGBM is not installed")
                model = lgb.LGBMRegressor(**params)
            
            # Define metrics for regression task
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from math import sqrt
            metrics = {
                'MSE': mean_squared_error,
                'RMSE': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error,
                'R-Squared': r2_score
            }
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results = {
            'model': model_name,
            'hyperparameters': params,
            'metrics': {}
        }
        
        for metric_name, metric_func in metrics.items():
            try:
                if task == 'classification' and metric_name in ['Precision', 'Recall', 'F1 Score']:
                    # Use average='binary' for binary classification, 'weighted' for multiclass
                    n_classes = len(set(y_train))
                    avg_method = 'binary' if n_classes == 2 else 'weighted'
                    score = metric_func(y_test, y_pred, average=avg_method)
                else:
                    score = metric_func(y_test, y_pred)
                results['metrics'][metric_name] = float(score)
            except Exception as e:
                print(f"Error calculating {metric_name}: {e}")
                results['metrics'][metric_name] = None
        # Store the trained model and training info for potential registration
        self.trained_model = model
        self.last_training_results = results
        self.last_training_method = model_training_method
        self.last_training_params = params
        
        return results

    def remove_features_by_threshold(self, feature_selection_method='pearson', threshold=0.5):
        """
        Remove features that don't meet the threshold criteria for the specified method.
        This creates a filtered dataset based on the feature selection results.
        
        Returns:
            dict: Information about removed features and updated dataset
        """
        try:
            print(f"Removing features with method: {feature_selection_method}, threshold: {threshold}")
            
            # First, get the feature selection results
            feature_results, _ = self.feature_selection(feature_selection_method, threshold)
            
            if not feature_results:
                return {'success': False, 'error': 'No feature selection results available'}
            
            # Determine which features meet the threshold
            if feature_selection_method in ['pearson', 'spearman']:
                # For correlation methods, use absolute value
                features_to_keep = [feature for feature, value in feature_results.items() 
                                if abs(value) >= threshold]
            else:  # feature_importance
                # For importance, use direct comparison
                features_to_keep = [feature for feature, value in feature_results.items() 
                                if value >= threshold]
            
            features_to_remove = [feature for feature in feature_results.keys() 
                                if feature not in features_to_keep]
            
            print(f"Features to keep: {features_to_keep}")
            print(f"Features to remove: {features_to_remove}")
            
            if not features_to_keep:
                return {
                    'success': False, 
                    'error': f'No features meet the threshold criteria of {threshold}'
                }
            
            # Check if data has been split
            if not hasattr(self, 'X_train') or self.X_train is None:
                return {
                    'success': False, 
                    'error': 'Data must be split first using Data Preparation'
                }
            
            # Apply feature removal to training and test sets
            # Only keep features that meet the threshold
            self.X_train = self.X_train[features_to_keep]
            self.X_test = self.X_test[features_to_keep]
            
            # Update the selected modeling columns to reflect the removal
            self._selected_modeling_columns = features_to_keep + [self.target] if hasattr(self, 'target') and self.target else features_to_keep
            self._save_conversions()
            
            print(f"Updated X_train shape: {self.X_train.shape}")
            print(f"Updated X_test shape: {self.X_test.shape}")
            
            return {
                'success': True,
                'features_kept': features_to_keep,
                'features_removed': features_to_remove,
                'method_used': feature_selection_method,
                'threshold_used': threshold,
                'new_shape': {
                    'X_train': list(self.X_train.shape),
                    'X_test': list(self.X_test.shape)
                },
                'feature_results': feature_results
            }
            
        except Exception as e:
            print(f"Error in remove_features_by_threshold: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def register_model(self, replace_existing=False):
        """
        Register the last trained model as the official model for performance testing
        
        Args:
            replace_existing (bool): If True, automatically replace any existing registered model
        """
        if not hasattr(self, 'trained_model') or self.trained_model is None:
            return {'success': False, 'error': 'No trained model found. Please train a model first.'}
        
        try:
            # Create model registry directory
            registry_dir = 'derivapro/static/model_registry'
            os.makedirs(registry_dir, exist_ok=True)
            
            # Check if there's an existing registration
            current_registration_file = os.path.join(registry_dir, 'current_registered_model.json')
            replaced_existing = False
            
            if os.path.exists(current_registration_file) and replace_existing:
                # Clean up existing registration first
                deregister_result = self.deregister_model()
                if deregister_result['success']:
                    replaced_existing = True
            
            # Generate timestamp for the registration
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save the registered model
            model_filename = f"registered_model_{timestamp}.pkl"
            model_path = os.path.join(registry_dir, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.trained_model, f)
            
            # Save model metadata
            metadata = {
                'registration_timestamp': timestamp,
                'model_type': self.last_training_method,
                'model_name': self.last_training_results['model'],
                'hyperparameters': self.last_training_params,
                'performance_metrics': self.last_training_results['metrics'],
                'target_variable': self.target,
                'task_type': self.task,
                'feature_columns': list(self.X_train.columns),
                'data_preprocessing': {
                    'conversions': getattr(self, '_conversions', {}),
                    'imputations': getattr(self, '_imputations', {}),
                    'normalizations': getattr(self, '_normalizations', {}),
                    'selected_columns': getattr(self, '_selected_modeling_columns', [])
                },
                'model_file': model_filename
            }
            
            metadata_filename = f"registered_model_metadata_{timestamp}.json"
            metadata_path = os.path.join(registry_dir, metadata_filename)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update the current registration pointer
            current_info = {
                'model_file': model_filename,
                'metadata_file': metadata_filename,
                'registration_timestamp': timestamp,
                'model_summary': f"{metadata['model_name']} ({metadata['model_type']})"
            }
            
            with open(current_registration_file, 'w') as f:
                json.dump(current_info, f, indent=2)
            
            # Store registration info in metadata for persistence
            self._registered_model_info = metadata
            self._save_conversions()  # This will include the registered model info
            
            message = f"Model '{metadata['model_name']}' has been successfully registered for performance testing"
            if replaced_existing:
                message = f"Model '{metadata['model_name']}' has been successfully registered, replacing the previous model"
            
            return {
                'success': True,
                'message': message,
                'replaced_existing': replaced_existing,
                'registration_details': {
                    'model_type': metadata['model_name'],
                    'timestamp': timestamp,
                    'performance_metrics': metadata['performance_metrics'],
                    'target_variable': metadata['target_variable']
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Registration failed: {str(e)}'}

    def get_registered_model_info(self):
        """
        Get information about the currently registered model
        """
        registry_dir = 'derivapro/static/model_registry'
        current_registration_file = os.path.join(registry_dir, 'current_registered_model.json')
        
        if not os.path.exists(current_registration_file):
            return {'success': False, 'message': 'No model is currently registered'}
        
        try:
            with open(current_registration_file, 'r') as f:
                current_info = json.load(f)
            
            # Load full metadata
            metadata_path = os.path.join(registry_dir, current_info['metadata_file'])
            with open(metadata_path, 'r') as f:
                full_metadata = json.load(f)
            
            return {
                'success': True,
                'registered_model': full_metadata,
                'summary': current_info['model_summary'],
                'registration_date': current_info['registration_timestamp']
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Failed to load registered model info: {str(e)}'}
    
    def deregister_model(self):
        """
        De-register the current model and clean up files
        """
        try:
            registry_dir = 'derivapro/static/model_registry'
            current_registration_file = os.path.join(registry_dir, 'current_registered_model.json')
            
            if os.path.exists(current_registration_file):
                # Load current registration info
                with open(current_registration_file, 'r') as f:
                    current_info = json.load(f)
                
                # Delete the registered model files
                model_file_path = os.path.join(registry_dir, current_info['model_file'])
                metadata_file_path = os.path.join(registry_dir, current_info['metadata_file'])
                
                files_deleted = []
                if os.path.exists(model_file_path):
                    os.remove(model_file_path)
                    files_deleted.append(current_info['model_file'])
                
                if os.path.exists(metadata_file_path):
                    os.remove(metadata_file_path)
                    files_deleted.append(current_info['metadata_file'])
                
                # Remove the current registration pointer
                os.remove(current_registration_file)
                
                # Clear registration info from metadata
                if hasattr(self, '_registered_model_info'):
                    delattr(self, '_registered_model_info')
                self._save_conversions()
                
                return {
                    'success': True,
                    'message': 'Model successfully de-registered',
                    'files_deleted': files_deleted
                }
            else:
                return {
                    'success': False,
                    'message': 'No model is currently registered'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'De-registration failed: {str(e)}'
            }

    def cleanup_temp_models(self):
        """
        Clean up temporary model files for this dataset
        """
        try:
            temp_dir = 'derivapro/static/temp_models'
            if not os.path.exists(temp_dir):
                return {'success': True, 'message': 'No temp files to clean'}
            
            dataset_name = os.path.basename(self.filepath).replace('.csv', '')
            files_deleted = []
            
            for filename in os.listdir(temp_dir):
                if filename.startswith(f'temp_model_{dataset_name}_'):
                    file_path = os.path.join(temp_dir, filename)
                    try:
                        os.remove(file_path)
                        files_deleted.append(filename)
                    except:
                        pass  # Continue if file can't be deleted
            
            return {
                'success': True,
                'message': f'Cleaned up {len(files_deleted)} temp files',
                'files_deleted': files_deleted
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Cleanup failed: {str(e)}'
            }
