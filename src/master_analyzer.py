import argparse
import os
from config_loader import load_config

from analysis_modules.gt_analysis.parser import BDDAnnotationParser 
from analysis_modules.gt_analysis.analyzer import GTMetricsAnalyzer
from analysis_modules.gt_analysis.visualizer import GTVisualizer
from analysis_modules.eval_analysis.analyzer import EvalRunner
from analysis_modules.eval_analysis.visualizer import EvalVisualizerRunner  

def main():
    parser = argparse.ArgumentParser(description="BDD Analysis Pipeline based on YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    cli_args = parser.parse_args()

    print(f"Loading configuration from: {cli_args.config}")
    cfg = load_config(cli_args.config)

    main_output_dir = cfg.get('project_paths', {}).get('main_output_dir', './analysis_run_outputs')
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"Main output directory: {main_output_dir}")

    # --- Ground Truth Analysis ---
    if cfg.get('run_modules', {}).get('ground_truth_analysis', False):
        print("\n=== Running Ground Truth Analysis (Placeholder) ===")
        gt_config = cfg.get('gt_analysis_config', {})

        parser_module = BDDAnnotationParser(gt_config, main_output_dir)

        cfg_project_paths = cfg.get("project_paths")
        bdd_labels_train_json_path = cfg_project_paths.get("bdd_labels_train_json")
        bdd_labels_val_json_path = cfg_project_paths.get("bdd_labels_val_json")
        print(f"Train JSON path: {bdd_labels_train_json_path}")
        print(f"val JSON PATH: {bdd_labels_val_json_path}")

        parsed_data_paths = parser_module.run(
            train_json_path_str= bdd_labels_train_json_path,
            val_json_path_str= bdd_labels_val_json_path)
        print()
        print(parsed_data_paths)
        print()
        print("************************ GT Analysis STARTED !!! ************************")
        print()
        
        gt_analyzer = GTMetricsAnalyzer(gt_config, cfg_project_paths,
                                         main_output_dir=main_output_dir)
        
        gt_analyzer()
        gt_visualizer = GTVisualizer(gt_config, main_output_dir)
        gt_visualizer()
        
        print()
        print("************************ GT Analysis COMPLETED !!! ************************")



    # --- Evaluation Analysis ---
    if cfg.get('run_modules', {}).get('evaluation_analysis', False):
        print("\n=== Running Evaluation Analysis ===")
        print()
        print("************************ EvalAnalysis COMPLETED !!! ************************")

        eval_config = cfg.get('eval_analysis_config', {})
        eval_runner = EvalRunner(cfg, eval_config, 
                                 cfg.get("project_paths"), 
                                 main_output_dir)

        eval_runner()
        eval_visualize = EvalVisualizerRunner(config=cfg, main_output_dir=main_output_dir)
        eval_visualize()

        print("************************ EvalAnalysis COMPLETED !!! ************************")



    # --- Qualitative Analysis ---
    if cfg.get('run_modules', {}).get('qualitative_analysis', False):
        print("\n=== Running Qualitative Analysis (Placeholder) ===")
        # TODO: Instantiate and call Qualitative analysis modules

    print("\nPipeline finished (skeleton run).")

if __name__ == "__main__":
    main()