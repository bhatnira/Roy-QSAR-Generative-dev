"""
Model-Agnostic QSAR Validation Pipeline
========================================

A completely model-agnostic and featurizer-agnostic pipeline that handles:
- Data leakage prevention
- Comprehensive validation
- All preprocessing steps

Users provide their own:
- Models (any sklearn-compatible estimator)
- Featurizers (any function that converts SMILES to features)
- Hyperparameters

The pipeline provides:
- Scaffold-based splitting (no leakage)
- Duplicate removal (no leakage)
- Proper scaling (no leakage)
- Cross-validation (no leakage)
- Comprehensive validation metrics
- Activity cliff detection
- Y-randomization tests
- Model complexity analysis
"""

import numpy as np
import pandas as pd
from typing import Callable, Any, Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler


class ModelAgnosticQSARPipeline:
    """
    A completely model-agnostic QSAR validation pipeline.
    
    Users provide their own models and featurizers.
    Pipeline provides data leakage prevention and validation.
    
    Parameters
    ----------
    featurizer : callable
        Function that converts SMILES to features.
        Signature: featurizer(smiles_list) -> np.ndarray
        Example: lambda smiles: get_morgan_fingerprints(smiles)
        
    model : sklearn-compatible estimator
        Any model with .fit() and .predict() methods.
        Examples: RandomForestRegressor(), Ridge(), XGBRegressor(), etc.
        
    smiles_col : str
        Name of column containing SMILES strings
        
    target_col : str
        Name of column containing target values
        
    validation_config : dict, optional
        Configuration for validation:
        - 'use_scaffold_split': bool (default True)
        - 'remove_duplicates': bool (default True)
        - 'scale_features': bool (default True)
        - 'detect_activity_cliffs': bool (default True)
        - 'run_y_randomization': bool (default True)
        - 'n_randomization_runs': int (default 10)
        - 'cv_folds': int (default 5)
        - 'test_size': float (default 0.2)
        - 'val_size': float (default 0.1)
    """
    
    def __init__(
        self,
        featurizer: Callable[[List[str]], np.ndarray],
        model: BaseEstimator,
        smiles_col: str = 'SMILES',
        target_col: str = 'Activity',
        validation_config: Optional[Dict[str, Any]] = None
    ):
        self.featurizer = featurizer
        self.model = model
        self.smiles_col = smiles_col
        self.target_col = target_col
        
        # Default validation configuration
        self.config = {
            'use_scaffold_split': True,
            'remove_duplicates': True,
            'scale_features': True,
            'detect_activity_cliffs': True,
            'run_y_randomization': True,
            'n_randomization_runs': 10,
            'cv_folds': 5,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42
        }
        if validation_config:
            self.config.update(validation_config)
        
        self.scaler = None
        self.results = {}
        
    def fit_predict_validate(
        self,
        df: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Complete pipeline: data preparation, training, validation.
        
        This function handles everything while preventing data leakage:
        1. Remove duplicates (BEFORE splitting)
        2. Scaffold-based splitting (zero scaffold overlap)
        3. Featurize molecules
        4. Scale features (using ONLY training statistics)
        5. Train model
        6. Comprehensive validation
        7. Activity cliff detection
        8. Y-randomization test
        9. Model complexity analysis
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset with SMILES and target columns
        verbose : bool
            Print progress messages
            
        Returns
        -------
        dict
            Complete validation results including:
            - 'performance': Test set metrics
            - 'cross_validation': CV metrics
            - 'y_randomization': Randomization test results
            - 'activity_cliffs': Detected activity cliffs
            - 'dataset_stats': Dataset statistics
            - 'data_leakage_checks': Leakage verification
        """
        if verbose:
            print("\n" + "="*70)
            print("MODEL-AGNOSTIC QSAR VALIDATION PIPELINE")
            print("="*70)
            print(f"Model: {type(self.model).__name__}")
            print(f"Featurizer: {self.featurizer.__name__ if hasattr(self.featurizer, '__name__') else 'Custom'}")
            print(f"Dataset size: {len(df)}")
            print("="*70 + "\n")
        
        # Import validation tools
        from . import (
            DatasetBiasAnalyzer,
            ActivityCliffDetector,
            ModelComplexityAnalyzer,
            PerformanceMetricsCalculator,
            YRandomizationTester,
            AssayNoiseEstimator
        )
        
        # Import leakage-free utilities
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from utils.qsar_utils_no_leakage import QSARDataProcessor, ScaffoldSplitter
        
        results = {}
        
        # ================================================================
        # STEP 1: DUPLICATE REMOVAL (BEFORE SPLITTING - NO LEAKAGE)
        # ================================================================
        if self.config['remove_duplicates']:
            if verbose:
                print("[STEP 1] Removing duplicates BEFORE splitting...")
            
            processor = QSARDataProcessor(self.smiles_col, self.target_col)
            df_clean = processor.remove_duplicates(df, method='smiles')
            
            n_removed = len(df) - len(df_clean)
            if verbose:
                print(f"  Removed {n_removed} duplicates ({n_removed/len(df)*100:.1f}%)")
                print(f"  Clean dataset: {len(df_clean)} compounds")
            
            results['duplicates_removed'] = n_removed
            df = df_clean.reset_index(drop=True)
        
        # ================================================================
        # STEP 2: SCAFFOLD-BASED SPLITTING (ZERO SCAFFOLD OVERLAP)
        # ================================================================
        if verbose:
            print("\n[STEP 2] Scaffold-based splitting (Bemis-Murcko)...")
        
        if self.config['use_scaffold_split']:
            splitter = ScaffoldSplitter(self.smiles_col)
            train_idx, val_idx, test_idx = splitter.scaffold_split(
                df,
                test_size=self.config['test_size'],
                val_size=self.config['val_size'],
                random_state=self.config['random_state']
            )
            
            # Verify zero scaffold overlap
            train_scaffolds = set(df.iloc[train_idx][splitter.scaffold_col])
            test_scaffolds = set(df.iloc[test_idx][splitter.scaffold_col])
            overlap = train_scaffolds & test_scaffolds
            
            if verbose:
                print(f"  Train: {len(train_idx)} compounds, {len(train_scaffolds)} scaffolds")
                print(f"  Val:   {len(val_idx)} compounds")
                print(f"  Test:  {len(test_idx)} compounds, {len(test_scaffolds)} scaffolds")
                print(f"  Scaffold overlap: {len(overlap)} (should be 0)")
                
                if len(overlap) > 0:
                    print(f"  [WARNING] Found scaffold overlap! Data leakage detected!")
        else:
            # Random split (not recommended but available)
            from sklearn.model_selection import train_test_split
            train_val_idx, test_idx = train_test_split(
                range(len(df)),
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=self.config['val_size']/(1-self.config['test_size']),
                random_state=self.config['random_state']
            )
            overlap = -1  # Not applicable for random split
        
        results['data_split'] = {
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx),
            'scaffold_overlap': len(overlap) if isinstance(overlap, set) else overlap,
            'split_method': 'scaffold' if self.config['use_scaffold_split'] else 'random'
        }
        
        # ================================================================
        # STEP 3: FEATURIZE MOLECULES
        # ================================================================
        if verbose:
            print("\n[STEP 3] Featurizing molecules...")
        
        smiles_train = df.iloc[train_idx][self.smiles_col].tolist()
        smiles_val = df.iloc[val_idx][self.smiles_col].tolist()
        smiles_test = df.iloc[test_idx][self.smiles_col].tolist()
        
        X_train = self.featurizer(smiles_train)
        X_val = self.featurizer(smiles_val)
        X_test = self.featurizer(smiles_test)
        
        y_train = df.iloc[train_idx][self.target_col].values
        y_val = df.iloc[val_idx][self.target_col].values
        y_test = df.iloc[test_idx][self.target_col].values
        
        if verbose:
            print(f"  Feature shape: {X_train.shape}")
            print(f"  Feature type: {type(X_train)}")
        
        results['features'] = {
            'n_features': X_train.shape[1],
            'feature_type': type(X_train).__name__
        }
        
        # ================================================================
        # STEP 4: SCALE FEATURES (USING ONLY TRAINING STATISTICS)
        # ================================================================
        if self.config['scale_features']:
            if verbose:
                print("\n[STEP 4] Scaling features (train statistics only)...")
            
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)  # Use train statistics
            X_test = self.scaler.transform(X_test)  # Use train statistics
            
            if verbose:
                print("  [OK] Features scaled using ONLY training statistics")
                print("  [OK] No information leakage from val/test sets")
        
        # ================================================================
        # STEP 5: TRAIN MODEL
        # ================================================================
        if verbose:
            print("\n[STEP 5] Training model...")
        
        self.model.fit(X_train, y_train)
        
        if verbose:
            print(f"  [OK] Model trained: {type(self.model).__name__}")
        
        # ================================================================
        # STEP 6: EVALUATE PERFORMANCE
        # ================================================================
        if verbose:
            print("\n[STEP 6] Evaluating performance...")
        
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)
        
        calculator = PerformanceMetricsCalculator()
        
        train_metrics = calculator.calculate_all_metrics(y_train, y_train_pred, name='Train')
        val_metrics = calculator.calculate_all_metrics(y_val, y_val_pred, name='Validation')
        test_metrics = calculator.calculate_all_metrics(y_test, y_test_pred, name='Test')
        
        if verbose:
            print(f"\n  Train:      R² = {train_metrics['r2']:.3f}, RMSE = {train_metrics['rmse']:.3f}")
            print(f"  Validation: R² = {val_metrics['r2']:.3f}, RMSE = {val_metrics['rmse']:.3f}")
            print(f"  Test:       R² = {test_metrics['r2']:.3f}, RMSE = {test_metrics['rmse']:.3f}")
        
        results['performance'] = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        # ================================================================
        # STEP 7: CROSS-VALIDATION (LEAKAGE-FREE)
        # ================================================================
        if verbose:
            print(f"\n[STEP 7] Cross-validation ({self.config['cv_folds']}-fold)...")
        
        # Use training data only for CV
        cv = KFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=self.config['random_state'])
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='r2')
        
        if verbose:
            print(f"  CV R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        results['cross_validation'] = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'cv_r2_scores': cv_scores.tolist()
        }
        
        # ================================================================
        # STEP 8: Y-RANDOMIZATION TEST (OVERFITTING CHECK)
        # ================================================================
        if self.config['run_y_randomization']:
            if verbose:
                print(f"\n[STEP 8] Y-randomization test ({self.config['n_randomization_runs']} runs)...")
            
            rand_tester = YRandomizationTester()
            rand_results = rand_tester.perform_y_randomization(
                X_train, y_train,
                self.model,
                n_iterations=self.config['n_randomization_runs'],
                cv_folds=self.config['cv_folds']
            )
            
            if verbose:
                print(f"  Random R² = {rand_results['r2_mean']:.3f} ± {rand_results['r2_std']:.3f}")
                print(f"  Real R² = {cv_scores.mean():.3f}")
                
                if rand_results['r2_mean'] > 0.2:
                    print("  [WARNING] High random R² - possible overfitting!")
                else:
                    print("  [OK] Low random R² - model is learning real patterns")
            
            results['y_randomization'] = rand_results
        
        # ================================================================
        # STEP 9: ACTIVITY CLIFF DETECTION
        # ================================================================
        if self.config['detect_activity_cliffs']:
            if verbose:
                print("\n[STEP 9] Detecting activity cliffs...")
            
            cliff_detector = ActivityCliffDetector(self.smiles_col, self.target_col)
            cliffs = cliff_detector.detect_activity_cliffs(
                df.iloc[train_idx],
                similarity_threshold=0.85,
                activity_threshold=2.0
            )
            
            if verbose:
                print(f"  Found {len(cliffs)} activity cliffs in training set")
                if len(cliffs) > 0:
                    print("  [INFO] Activity cliffs indicate SAR discontinuities")
            
            results['activity_cliffs'] = cliffs
        
        # ================================================================
        # STEP 10: DATASET ANALYSIS
        # ================================================================
        if verbose:
            print("\n[STEP 10] Dataset analysis...")
        
        analyzer = DatasetBiasAnalyzer(self.smiles_col, self.target_col)
        scaffold_diversity = analyzer.analyze_scaffold_diversity(df)
        activity_dist = analyzer.analyze_activity_distribution(df)
        
        if verbose:
            print(f"  Unique scaffolds: {scaffold_diversity['n_scaffolds']}")
            print(f"  Activity range: [{activity_dist['min']:.2f}, {activity_dist['max']:.2f}]")
        
        results['dataset_stats'] = {
            'scaffold_diversity': scaffold_diversity,
            'activity_distribution': activity_dist
        }
        
        # ================================================================
        # STEP 11: MODEL COMPLEXITY ANALYSIS
        # ================================================================
        if verbose:
            print("\n[STEP 11] Model complexity analysis...")
        
        complexity_results = ModelComplexityAnalyzer.analyze_complexity(
            n_samples=len(train_idx),
            n_features=X_train.shape[1],
            model_type=type(self.model).__name__.lower()
        )
        
        if verbose:
            print(f"  Samples/features ratio: {len(train_idx)/X_train.shape[1]:.2f}")
            print(f"  Status: {complexity_results['status']}")
        
        results['model_complexity'] = complexity_results
        
        # ================================================================
        # STEP 12: ESTIMATE EXPERIMENTAL ERROR
        # ================================================================
        if verbose:
            print("\n[STEP 12] Estimating experimental error...")
        
        noise_estimator = AssayNoiseEstimator()
        exp_error = noise_estimator.estimate_experimental_error(df, self.target_col)
        
        if verbose:
            print(f"  Estimated experimental error: {exp_error['experimental_error']:.3f}")
            print(f"  Test RMSE: {test_metrics['rmse']:.3f}")
            
            if test_metrics['rmse'] < exp_error['experimental_error'] * 0.6:
                print("  [WARNING] RMSE suspiciously low - check for data leakage!")
            else:
                print("  [OK] RMSE is reasonable given experimental noise")
        
        results['experimental_error'] = exp_error
        
        # ================================================================
        # STEP 13: DATA LEAKAGE VERIFICATION
        # ================================================================
        if verbose:
            print("\n[STEP 13] Data leakage verification...")
        
        leakage_checks = {
            'duplicates_removed_before_split': self.config['remove_duplicates'],
            'scaffold_split_used': self.config['use_scaffold_split'],
            'scaffold_overlap': results['data_split']['scaffold_overlap'],
            'scaler_fit_on_train_only': self.config['scale_features'],
            'test_rmse_vs_exp_error': test_metrics['rmse'] / exp_error['experimental_error']
        }
        
        all_checks_pass = (
            leakage_checks['duplicates_removed_before_split'] and
            (leakage_checks['scaffold_overlap'] == 0 if self.config['use_scaffold_split'] else True) and
            leakage_checks['scaler_fit_on_train_only'] and
            leakage_checks['test_rmse_vs_exp_error'] >= 0.6
        )
        
        if verbose:
            print("  Leakage Prevention Checks:")
            print(f"    [{'OK' if leakage_checks['duplicates_removed_before_split'] else 'FAIL'}] Duplicates removed before split")
            print(f"    [{'OK' if leakage_checks['scaffold_split_used'] else 'SKIP'}] Scaffold-based split used")
            if self.config['use_scaffold_split']:
                print(f"    [{'OK' if leakage_checks['scaffold_overlap'] == 0 else 'FAIL'}] Zero scaffold overlap")
            print(f"    [{'OK' if leakage_checks['scaler_fit_on_train_only'] else 'SKIP'}] Scaler fit on train only")
            print(f"    [{'OK' if leakage_checks['test_rmse_vs_exp_error'] >= 0.6 else 'WARNING'}] Test RMSE vs experimental error")
            print(f"\n  Overall: [{'OK' if all_checks_pass else 'CHECK WARNINGS'}] Data leakage prevention")
        
        results['data_leakage_checks'] = leakage_checks
        
        # ================================================================
        # SUMMARY
        # ================================================================
        if verbose:
            print("\n" + "="*70)
            print("VALIDATION COMPLETE")
            print("="*70)
            print(f"Test R²:  {test_metrics['r2']:.3f}")
            print(f"Test RMSE: {test_metrics['rmse']:.3f}")
            print(f"Data leakage prevention: [{'OK' if all_checks_pass else 'CHECK WARNINGS'}]")
            print("="*70 + "\n")
        
        self.results = results
        return results
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """
        Make predictions on new SMILES.
        
        Parameters
        ----------
        smiles_list : list of str
            List of SMILES strings
            
        Returns
        -------
        np.ndarray
            Predicted values
        """
        X = self.featurizer(smiles_list)
        
        if self.config['scale_features'] and self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get a summary of validation results as a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Summary of all validation metrics
        """
        if not self.results:
            raise ValueError("No results available. Run fit_predict_validate() first.")
        
        summary_data = []
        
        # Performance metrics
        for split in ['train', 'validation', 'test']:
            metrics = self.results['performance'][split]
            summary_data.append({
                'Category': 'Performance',
                'Metric': f'{split.capitalize()} R²',
                'Value': f"{metrics['r2']:.3f}"
            })
            summary_data.append({
                'Category': 'Performance',
                'Metric': f'{split.capitalize()} RMSE',
                'Value': f"{metrics['rmse']:.3f}"
            })
        
        # Cross-validation
        cv = self.results['cross_validation']
        summary_data.append({
            'Category': 'Cross-Validation',
            'Metric': 'CV R² (mean ± std)',
            'Value': f"{cv['cv_r2_mean']:.3f} ± {cv['cv_r2_std']:.3f}"
        })
        
        # Y-randomization
        if 'y_randomization' in self.results:
            rand = self.results['y_randomization']
            summary_data.append({
                'Category': 'Y-Randomization',
                'Metric': 'Random R² (mean ± std)',
                'Value': f"{rand['r2_mean']:.3f} ± {rand['r2_std']:.3f}"
            })
        
        # Dataset stats
        stats = self.results['dataset_stats']
        summary_data.append({
            'Category': 'Dataset',
            'Metric': 'N compounds',
            'Value': str(stats['scaffold_diversity']['n_compounds'])
        })
        summary_data.append({
            'Category': 'Dataset',
            'Metric': 'N scaffolds',
            'Value': str(stats['scaffold_diversity']['n_scaffolds'])
        })
        
        # Data split
        split_info = self.results['data_split']
        summary_data.append({
            'Category': 'Data Split',
            'Metric': 'Train/Val/Test',
            'Value': f"{split_info['n_train']}/{split_info['n_val']}/{split_info['n_test']}"
        })
        summary_data.append({
            'Category': 'Data Split',
            'Metric': 'Scaffold overlap',
            'Value': str(split_info['scaffold_overlap'])
        })
        
        return pd.DataFrame(summary_data)
