from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify
import os  # ← Add this here
import pandas as pd
import pickle  # ← Add this here
from werkzeug.utils import secure_filename
from ..models.mdls_prepayment_v2 import PrepaymentDataUploader, Validation


prepayment_v2_bp = Blueprint('prepayment_v2', __name__)
UPLOAD_FOLDER = 'derivapro/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@prepayment_v2_bp.route('/prepayment-model-validator', methods=['GET', 'POST'])
def prepayment_model_validator():
    uploader = PrepaymentDataUploader(upload_folder=UPLOAD_FOLDER)
    upload_success, uploaded_filename, data_info = False, None, None
    summary_num, summary_cat, missing_table = None, None, None
    dist_plot, scatter_plot, heatmap_plot = None, None, None
    feature_selection_plot, feature_selection_result = None, None
    df_columns = []

    if request.method == 'POST':
        if 'file' in request.files:
            result = uploader.save_uploaded_file(request.files['file'])
            if result['success']:
                session['uploaded_data_file_path'] = result['filepath']
                session['uploaded_filename'] = result['filename']
                flash(f'File "{result["filename"]}" uploaded successfully!', 'success')
            else:
                flash(result['error'], 'error')
                return redirect(request.url)
        else:
            flash('No file selected', 'error')
            return redirect(request.url)

    # Load existing file if present
    if 'uploaded_data_file_path' in session and os.path.exists(session['uploaded_data_file_path']):
        filepath = session['uploaded_data_file_path']
        uploaded_filename = session['uploaded_filename']
        validator = Validation(filepath)
        df_columns = list(validator.data.columns)

        # Summaries
        summary_num = validator.summary_numerical()
        summary_cat = validator.summary_categorical()
        missing_table = validator.missing_values()

        # Handle plotting requests
        if request.args.get('dist_col'):
            dist_plot = validator.plot_distribution(request.args['dist_col'])
        if request.args.get('scatter_x') and request.args.get('scatter_y'):
            scatter_plot = validator.plot_scatter(request.args['scatter_x'], request.args['scatter_y'])
        heatmap_plot = validator.plot_heatmap()

        # Clear feature selection plots from session on fresh page load
        feature_selection_plot = session.get('feature_selection_plot')
        feature_selection_result = session.get('feature_selection_result')

        # Only clear session on fresh page load (not from feature selection reload)
        if request.method == 'GET' and not request.args and not feature_selection_plot:
            # This means it's a true fresh page load, not a reload after feature selection
            session.pop('feature_selection_plot', None)
            session.pop('feature_selection_result', None)
            feature_selection_plot = None
            feature_selection_result = None

        upload_success = True
        data_info = {
            'filename': uploaded_filename,
            'shape': validator.data.shape,
            'columns': df_columns,
            'head': validator.data.head().to_html(classes='table table-striped')
        }

    return render_template(
        'prepayment_v2.html',
        upload_success=upload_success,
        uploaded_filename=uploaded_filename,
        data_info=data_info,
        summary_num=summary_num,
        summary_cat=summary_cat,
        missing_table=missing_table,
        dist_plot=dist_plot,
        scatter_plot=scatter_plot,
        heatmap_plot=heatmap_plot,
        feature_selection_plot=feature_selection_plot,
        feature_selection_result=feature_selection_result,
        df_columns=df_columns
    )

@prepayment_v2_bp.route('/prepayment-model-validator/convert_columns', methods=['POST'])
def convert_columns():
    """Handle column type conversions"""
    try:
        # Check if file is uploaded
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Uploaded file not found'})
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})
        
        columns_to_convert = data.get('columns', [])
        conversion_type = data.get('type', 'numeric')
        
        if not columns_to_convert:
            return jsonify({'success': False, 'error': 'No columns specified'})
        
        # Create validator and convert
        validator = Validation(filepath)
        
        # Check if column exists
        column_name = columns_to_convert[0]
        if column_name not in validator.data.columns:
            return jsonify({'success': False, 'error': f'Column "{column_name}" not found'})
        
        # Perform conversion
        validator.convert_columns(columns_to_convert, conversion_type)

        # Save the updated data back to the file
        validator.data.to_csv(filepath, index=False)

        return jsonify({'success': True, 'message': f'Successfully converted "{column_name}" to {conversion_type}'})
        
    except Exception as e:
        # Log the actual error for debugging
        import traceback
        print(f"Error in convert_columns: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Conversion failed: {str(e)}'})

@prepayment_v2_bp.route('/prepayment-model-validator/impute_data', methods=['POST'])
def impute_data():
    """Handle data imputation"""
    try:
        # Check if file is uploaded
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Uploaded file not found'})
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})
        
        columns_to_impute = data.get('columns', [])
        imputation_method = data.get('method', 'mean')
        
        # Create validator and impute
        validator = Validation(filepath)
        
        # Check if column exists
        column_name = columns_to_impute[0]
        if column_name not in validator.data.columns:
            return jsonify({'success': False, 'error': f'Column "{column_name}" not found'})
        
        # Perform imputation
        validator.impute_data(columns_to_impute, imputation_method)
        
        # Save the updated data back to the file
        validator.data.to_csv(filepath, index=False)
        
        return jsonify({'success': True, 'message': f'Successfully imputed "{column_name}" with {imputation_method}'})
        
    except Exception as e:
        # Log the actual error for debugging
        import traceback
        print(f"Error in impute_data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Imputation failed: {str(e)}'})

@prepayment_v2_bp.route('/prepayment-model-validator/normalize_data', methods=['POST'])
def normalize_data():
    """Handle data normalization"""
    try:
        # Check if file is uploaded
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Uploaded file not found'})
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})
        
        columns_to_normalize = data.get('columns', [])
        normalization_method = data.get('method', 'standard')
        
        # Create validator and impute
        validator = Validation(filepath)
        
        # Check if column exists
        column_name = columns_to_normalize[0]
        if column_name not in validator.data.columns:
            return jsonify({'success': False, 'error': f'Column "{column_name}" not found'})
        
        # Perform imputation
        validator.normalize_data(columns_to_normalize, normalization_method)
        
        # Save the updated data back to the file
        validator.data.to_csv(filepath, index=False)
        
        return jsonify({'success': True, 'message': f'Successfully normalized "{column_name}" with {normalization_method}'})
        
    except Exception as e:
        # Log the actual error for debugging
        import traceback
        print(f"Error in normalize_data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Normalization failed: {str(e)}'})

@prepayment_v2_bp.route('/prepayment-model-validator/prepare_data', methods=['POST'])
def prepare_data():
    """Handle data preparation"""
    try:
        # Check if file is uploaded
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Uploaded file not found'})
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})
        # Create validator
        validator = Validation(filepath)
        
        columns_for_target = data.get('columns', [])
        # Check if column exists
        column_name = columns_for_target[0]
        if column_name not in validator.data.columns:
            return jsonify({'success': False, 'error': f'Column "{column_name}" not found'})
        
        
        
                
        target_variable = column_name
        task = data.get('task', 'classification')
        train_test_split_ratio = float(data.get('train_test_split_ratio', 0.2))  # Convert to float with default
        random_state = int(data.get('random_state', 42))  # Convert to int with default

        # Perform Data Split
        split_success = validator.train_test_split_data(target_variable, task, train_test_split_ratio, random_state)
        
        if not split_success:
            return jsonify({'success': False, 'error': 'Failed to split data'})
        
        # Print simple summary for debugging
        
        print(f"Data split completed:")
        print(f"  Training set: {validator.X_train.shape[0]} rows, {validator.X_train.shape[1]} columns")
        print(f"  Test set: {validator.X_test.shape[0]} rows, {validator.X_test.shape[1]} columns")
        print(f"  Task: {task}")
        print(f"  Target variable: {target_variable}")
        print(f"  Target data type: {validator.data[target_variable].dtype}")
        print(f"  Unique target values: {len(validator.data[target_variable].unique())}")
        if task == 'classification':
            print(f"  Class distribution: {dict(validator.data[target_variable].value_counts())}")
            print(f"  Stratification applied: Yes")
        else:
            print(f"  Target range: {validator.data[target_variable].min()} to {validator.data[target_variable].max()}")
            print(f"  Stratification applied: No")
        

        return jsonify({
            'success': True, 
            'message': f'Successfully split data into training and testing sets',
            'split_info': {
                'train_rows': validator.X_train.shape[0],
                'train_cols': validator.X_train.shape[1],
                'test_rows': validator.X_test.shape[0],
                'test_cols': validator.X_test.shape[1]
            }
        })
    
    except Exception as e:
        # Log the actual error for debugging
        import traceback
        print(f"Error in prepare_data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Data split failed: {str(e)}'})   
        
@prepayment_v2_bp.route('/prepayment-model-validator/get_columns', methods=['GET'])
def get_columns():
    """Get available columns for column selection"""
    try:
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Uploaded file not found'})
        
        validator = Validation(filepath)
        columns = validator.data.columns.tolist()
        
        # Get target variable if it exists
        target_variable = getattr(validator, 'target', None)
        
        return jsonify({
            'success': True, 
            'columns': columns,
            'target_variable': target_variable
        })
        
    except Exception as e:
        import traceback
        print(f"Error in get_columns: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Failed to get columns: {str(e)}'})

@prepayment_v2_bp.route('/prepayment-model-validator/get_current_features', methods=['GET'])
def get_current_features():
    """Get current available features after feature selection and threshold validation"""
    try:
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Uploaded file not found'})
        
        validator = Validation(filepath)
        
        # Check if data has been prepared and features selected
        if not hasattr(validator, 'X_train') or validator.X_train is None:
            return jsonify({'success': False, 'error': 'Please prepare data and run feature selection first'})
        
        # Get current columns from X_train (after feature selection/threshold validation)
        current_features = validator.X_train.columns.tolist()
        target_variable = getattr(validator, 'target', None)
        
        print(f"Current features available: {len(current_features)} - {current_features}")
        
        return jsonify({
            'success': True, 
            'features': current_features,
            'feature_count': len(current_features),
            'target_variable': target_variable,
            'dataset_shape': {
                'X_train': list(validator.X_train.shape),
                'X_test': list(validator.X_test.shape)
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Error in get_current_features: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Failed to get current features: {str(e)}'})


@prepayment_v2_bp.route('/prepayment-model-validator/apply_column_selection', methods=['POST'])
def apply_column_selection():
    """Apply column selection for modeling"""
    try:
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Uploaded file not found'})
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})
        
        selected_columns = data.get('selected_columns', [])
        
        if not selected_columns:
            return jsonify({'success': False, 'error': 'No columns selected'})
        
        validator = Validation(filepath)
        
        # Check if data has been prepared (split) first
        if not hasattr(validator, 'X_train') or validator.X_train is None:
            return jsonify({'success': False, 'error': 'Please prepare data first (Data Preparation step)'})
        
        # Apply column selection using the method
        validator.apply_column_selection(selected_columns)
        
        return jsonify({
            'success': True, 
            'message': f'Column selection applied. {len(selected_columns)} columns selected for modeling.',
            'selected_columns': selected_columns,
            'new_shape': {
                'X_train': list(validator.X_train.shape),
                'X_test': list(validator.X_test.shape)
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Error in apply_column_selection: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Column selection failed: {str(e)}'})

@prepayment_v2_bp.route('/prepayment-model-validator/feature_selection', methods=['POST'])
def feature_selection():
    """Handle feature selection"""
    try:       
        # Check if file is uploaded
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Uploaded file not found'})
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})

        # Create validator
        validator = Validation(filepath)
        
        # Check if data has been split (required for feature selection)
        if not hasattr(validator, 'target') or not hasattr(validator, 'X_train'):
            return jsonify({'success': False, 'error': 'Please prepare data first by splitting into train/test sets'})
        
        # Perform feature selection
        feature_selection_method = data.get('method', 'pearson')
        threshold = float(data.get('threshold', 0.1))
        
        print(f"Feature selection request: method={feature_selection_method}, threshold={threshold}")
        print(f"Validator target: {getattr(validator, 'target', 'Not set')}")
        print(f"Data shape: {validator.data.shape}")
        
        feature_selection_result, feature_selection_plot = validator.feature_selection(feature_selection_method, threshold)
        
        print(f"Feature selection result type: {type(feature_selection_result)}")
        print(f"Plot generated: {feature_selection_plot is not None}")
        
        if feature_selection_result is None:
            return jsonify({'success': False, 'error': 'Feature selection failed - no results returned'})

        if feature_selection_plot is None:
            return jsonify({'success': False, 'error': 'Feature selection failed - no plot generated'})

        # Store the plot and results in session so they can be displayed
        # Store only the plot file path in session (much smaller)
        session['feature_selection_plot'] = feature_selection_plot  # This is now just a file path
        session['feature_selection_result'] = feature_selection_result
        session['feature_selection_method'] = feature_selection_method
        session['feature_selection_threshold'] = threshold

        return jsonify({
            'success': True, 
            'message': f'Successfully performed {feature_selection_method} correlation analysis with threshold {threshold}',
            'feature_selection_result': feature_selection_result,
            'feature_selection_plot': feature_selection_plot
        })
    
    except Exception as e:
        # Log the actual error for debugging
        import traceback
        print(f"Error in feature_selection: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Feature selection failed: {str(e)}'})


@prepayment_v2_bp.route('/prepayment-model-validator/remove_features', methods=['POST'])
def remove_features():
    """Handle feature removal based on threshold criteria"""
    try:
        # Check if file is uploaded
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Uploaded file not found'})
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})

        # Create validator
        validator = Validation(filepath)
        
        # Check if data has been split (required for feature removal)
        if not hasattr(validator, 'target') or not hasattr(validator, 'X_train'):
            return jsonify({'success': False, 'error': 'Please prepare data first by splitting into train/test sets'})
        
        # Perform feature removal
        feature_selection_method = data.get('method', 'pearson')
        threshold = float(data.get('threshold', 0.1))
        
        print(f"Feature removal request: method={feature_selection_method}, threshold={threshold}")
        
        removal_result = validator.remove_features_by_threshold(feature_selection_method, threshold)
        
        if not removal_result['success']:
            return jsonify({'success': False, 'error': removal_result['error']})

        return jsonify({
            'success': True, 
            'message': f'Successfully removed {len(removal_result["features_removed"])} features using {feature_selection_method} with threshold {threshold}',
            'removal_result': removal_result
        })
    
    except Exception as e:
        # Log the actual error for debugging
        import traceback
        print(f"Error in remove_features: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Feature removal failed: {str(e)}'})


@prepayment_v2_bp.route('/prepayment-model-validator/final_column_selection', methods=['POST'])
def final_column_selection():
    """Handle final column selection"""
    try:
        # Check if file is uploaded
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})

        filepath = session['uploaded_data_file_path']
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Uploaded file not found'})

        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})

        # Get selected columns and validate
        selected_columns = data.get('selected_columns', [])
        if not selected_columns:
            return jsonify({'success': False, 'error': 'No columns selected'})

        # Create validator
        validator = Validation(filepath)
        
        # Check if data has been split (required for final column selection)
        if not hasattr(validator, 'target') or not hasattr(validator, 'X_train'):
            return jsonify({'success': False, 'error': 'Please prepare data first by splitting into train/test sets'})
        
        print(f"Final column selection request: {len(selected_columns)} columns selected")
        print(f"Selected columns: {selected_columns}")
        
        # Perform final column selection
        result = validator.final_column_selection(selected_columns)

        return jsonify({
            'success': True, 
            'message': f'Successfully applied final column selection. Using {len(selected_columns)} columns for modeling.',
            'selected_columns': selected_columns,
            'new_shape': {
                'X_train': list(validator.X_train.shape),
                'X_test': list(validator.X_test.shape)
            }
        })
    
    except Exception as e:
        # Log the actual error for debugging
        import traceback
        print(f"Error in final_column_selection: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Final column selection failed: {str(e)}'})

@prepayment_v2_bp.route('/prepayment-model-validator/model_training', methods=['POST'])
def model_training():
    """Handle model training"""
    import os
    import pickle
    try:
        # Check if file is uploaded
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Uploaded file not found'})
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})
        
        # Create validator
        validator = Validation(filepath)
        
        # Check if data has been split (required for model training)
        if not hasattr(validator, 'target') or not hasattr(validator, 'X_train'):
            return jsonify({'success': False, 'error': 'Please prepare data first by splitting into train/test sets'})
        
        # Perform model training
        model_training_method = data.get('method', 'random_forest')
        hyperparameters = data.get('hyperparameters', {})
        
        # TRAIN THE MODEL FIRST
        model_training_result = validator.model_training(model_training_method, hyperparameters)
        
        # THEN store in temp file if successful (NOT session)
        if model_training_result:
            # Store the trained model in temp file for registration
            import uuid
            
            # Create temp directory
            temp_dir = 'derivapro/static/temp_models'
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create UNIQUE filename with timestamp or UUID
            dataset_name = os.path.basename(filepath).replace('.csv', '')
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_model_file = os.path.join(temp_dir, f'temp_model_{dataset_name}_{timestamp}.pkl')
            
            model_data = {
                'model': validator.trained_model,
                'results': validator.last_training_results,
                'method': validator.last_training_method,
                'params': validator.last_training_params
            }
            
            # Save to file instead of session
            with open(temp_model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Store only the filename in session (small)
            session['last_trained_model_file'] = temp_model_file
            print(f"✅ Stored trained model in temp file: {temp_model_file}")
            
            # Store only the filename in session (small)
            session['last_trained_model_file'] = temp_model_file
            print(f"✅ Stored trained model in temp file: {temp_model_file}")
        
        return jsonify({'success': True, 'message': f'Successfully trained {model_training_method} model', 'model_training_result': model_training_result})
        
    except Exception as e:
        # Log the actual error for debugging
        import traceback
        print(f"Error in model_training: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Model training failed: {str(e)}'})

@prepayment_v2_bp.route('/prepayment-model-validator/register_model', methods=['POST'])
def register_model():
    """Register the trained model for performance testing"""
    try:
        print("🔄 Starting model registration...")
        
        # Check if file is uploaded
        if 'uploaded_data_file_path' not in session:
            print("❌ No file uploaded in session")
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        print(f"📁 Using filepath: {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return jsonify({'success': False, 'error': 'Uploaded file not found'})
        
        # Create validator
        validator = Validation(filepath)
        print("✅ Validator created successfully")
        
        # Check if there's a trained model file in session
        if 'last_trained_model_file' not in session:
            print("❌ No trained model file in session")
            return jsonify({'success': False, 'error': 'No trained model found. Please train a model first.'})
        
        temp_model_file = session['last_trained_model_file']
        print(f"🔍 Looking for temp model file: {temp_model_file}")
        
        # Check if temp file exists
        if not os.path.exists(temp_model_file):
            print(f"❌ Temp model file not found: {temp_model_file}")
            
            # Let's see what files DO exist in the temp directory
            temp_dir = 'derivapro/static/temp_models'
            if os.path.exists(temp_dir):
                existing_files = os.listdir(temp_dir)
                print(f"📂 Files in temp directory: {existing_files}")
            else:
                print("📂 Temp directory doesn't exist")
                
            return jsonify({'success': False, 'error': 'Trained model file not found. Please train again.'})
        
        print(f"✅ Temp model file exists: {temp_model_file}")
        
        # Load the trained model from temp file
        try:
            print(f"📖 Loading model from: {temp_model_file}")
            with open(temp_model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            validator.trained_model = model_data['model']
            validator.last_training_results = model_data['results']
            validator.last_training_method = model_data['method']
            validator.last_training_params = model_data['params']
            print("✅ Restored trained model from temp file")
        except Exception as e:
            print(f"❌ Failed to load model from temp file: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': 'Failed to load trained model. Please train again.'})
        
        # Register the model
        print("🚀 Calling validator.register_model()...")
        result = validator.register_model()
        
        if result['success']:
            print(f"✅ Registration successful! Details: {result['registration_details']}")
            print("✅ Model registered successfully")
        else:
            print(f"❌ Registration failed: {result['error']}")
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        print(f"💥 Error in register_model: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Model registration failed: {str(e)}'})

@prepayment_v2_bp.route('/prepayment-model-validator/get_registered_model', methods=['GET'])
def get_registered_model():
    """Get information about the currently registered model"""
    try:
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Uploaded file not found'})
        
        validator = Validation(filepath)
        result = validator.get_registered_model_info()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to get registered model info: {str(e)}'})


@prepayment_v2_bp.route('/prepayment-model-validator/retrain_model', methods=['POST'])
def retrain_model():
    """Prepare for retraining without clearing session"""
    try:
        print("🔄 Ready for new training...")
        
        # Check if file is uploaded
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        # DON'T clear session - let the temp file remain available for re-registration
        # Only clear it when a new model is actually trained
        print("🧹 Training state ready (session preserved)")
        
        return jsonify({
            'success': True,
            'message': 'Ready for new training. You can still re-register the current model if needed.',
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Preparation failed: {str(e)}'})

@prepayment_v2_bp.route('/prepayment-model-validator/reregister_model', methods=['POST'])
def reregister_model():
    """Replace the current registered model with the newly trained one"""
    try:
        print("🔄 Starting model re-registration...")
        
        # Check if file is uploaded
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        validator = Validation(filepath)
        
        # Check if there's a newly trained model
        if 'last_trained_model_file' not in session:
            return jsonify({'success': False, 'error': 'No newly trained model found. Please train a model first.'})
        
        temp_model_file = session['last_trained_model_file']
        
        if not os.path.exists(temp_model_file):
            return jsonify({'success': False, 'error': 'Newly trained model file not found. Please train again.'})
        
        # Load the new model
        with open(temp_model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        validator.trained_model = model_data['model']
        validator.last_training_results = model_data['results']
        validator.last_training_method = model_data['method']
        validator.last_training_params = model_data['params']
        
        # Register the new model (this will automatically replace any existing registration)
        result = validator.register_model(replace_existing=True)
        
        return jsonify({
            'success': result['success'],
            'message': 'Model successfully registered, replacing any previous registration.',
            'registration_details': result.get('registration_details', {}),
            'replaced_existing': result.get('replaced_existing', False)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Model re-registration failed: {str(e)}'})

@prepayment_v2_bp.route('/prepayment-model-validator/deregister_model', methods=['POST'])
def deregister_model():
    """Deregister the currently registered model"""
    try:
        print("🔄 Starting model deregistration...")
        
        # Check if file is uploaded
        if 'uploaded_data_file_path' not in session:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        filepath = session['uploaded_data_file_path']
        validator = Validation(filepath)
        
        # Deregister the model
        result = validator.deregister_model()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Model deregistration failed: {str(e)}'})



@prepayment_v2_bp.route('/delete_upload', methods=['POST'])
def delete_upload():
    if 'uploaded_data_file_path' in session:
        filepath = session['uploaded_data_file_path']
        if os.path.exists(filepath):
            os.remove(filepath)
        session.pop('uploaded_data_file_path', None)
        session.pop('uploaded_filename', None)
        flash('File deleted successfully', 'success')
    return redirect(url_for('prepayment_v2.prepayment_model_validator'))

@prepayment_v2_bp.route('/prepayment-model-validator/start_over', methods=['POST'])
def start_over():
    """Complete reset - delete all files, metadata, and session data"""
    try:
        cleanup_results = []
        
        # 1. Clean up model files FIRST (while CSV file still exists)
        if 'uploaded_data_file_path' in session:
            filepath = session['uploaded_data_file_path']
            if os.path.exists(filepath):
                try:
                    validator = Validation(filepath)
                    
                    # Clean up temporary model files
                    temp_cleanup = validator.cleanup_temp_models()
                    if temp_cleanup['success'] and temp_cleanup.get('files_deleted'):
                        cleanup_results.append(f"✓ Deleted {len(temp_cleanup['files_deleted'])} temporary model file(s)")
                    
                    # Deregister and delete registered model files
                    deregister_result = validator.deregister_model()
                    if deregister_result['success'] and deregister_result.get('files_deleted'):
                        cleanup_results.append(f"✓ Deleted {len(deregister_result['files_deleted'])} registered model file(s)")
                    elif deregister_result['success']:
                        cleanup_results.append("✓ No registered model to remove")
                        
                except Exception as e:
                    print(f"Warning: Could not clean up model files: {e}")
                    cleanup_results.append(f"⚠️ Model cleanup warning: {str(e)}")
        
        # 2. Delete uploaded CSV file and its metadata
        if 'uploaded_data_file_path' in session:
            filepath = session['uploaded_data_file_path']
            if os.path.exists(filepath):
                # Delete the CSV file
                os.remove(filepath)
                cleanup_results.append(f"✓ Deleted CSV file: {os.path.basename(filepath)}")
                
                # Delete metadata file
                metadata_file = filepath.replace('.csv', '_metadata.json')
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                    cleanup_results.append(f"✓ Deleted metadata file: {os.path.basename(metadata_file)}")
        
        # 3. Delete feature selection plot files
        # plots_dir = 'derivapro/static/plots'
        plots_dir = os.path.join(current_app.static_folder, 'plots')  # ← Change this line
        if os.path.exists(plots_dir):
            plot_files_deleted = 0
            for filename in os.listdir(plots_dir):
                if any(plot_type in filename for plot_type in ['feature_selection', 'spearman_correlation', 'feature_importance']):
                    try:
                        os.remove(os.path.join(plots_dir, filename))
                        plot_files_deleted += 1
                    except:
                        pass  # Continue even if some files can't be deleted
            if plot_files_deleted > 0:
                cleanup_results.append(f"✓ Deleted {plot_files_deleted} plot file(s)")
        
        # 4. Clear all session variables related to prepayment analysis
        session_keys_to_clear = [
            'uploaded_data_file_path',
            'uploaded_filename', 
            'feature_selection_plot',
            'feature_selection_result',
            'feature_selection_method',
            'feature_selection_threshold',
            'preserve_feature_plots',
            'last_trained_model_file'
        ]
        
        cleared_sessions = 0
        for key in session_keys_to_clear:
            if key in session:
                session.pop(key, None)
                cleared_sessions += 1
        
        if cleared_sessions > 0:
            cleanup_results.append(f"✓ Cleared {cleared_sessions} session variable(s)")

        # 5. Final cleanup summary
        if cleanup_results:
            cleanup_results.append("✓ All data and metadata successfully cleared")
        else:
            cleanup_results.append("✓ No data to clear - already clean")
            
        return jsonify({
            'success': True,
            'message': 'Successfully started over - all data cleared',
            'cleanup_details': cleanup_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error during cleanup: {str(e)}'
        })