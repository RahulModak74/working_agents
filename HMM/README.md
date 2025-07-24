HMM analyser for EDR

1. Create synthetic data: create_detailed_defender_crowdstrike_synethetic_data.py

2. Convert to acceptable format: convert_defender_crowdstrike_to_model_log_format.py
3. Verfiy
4. Run pyro based analysis: fixed_pyro_2.py
5. See attack chains: python3 standalone_attack_chain_analyzer.py     --pyro-results enhanced_pyro_bayesian_analysis.csv    --original-data synthetic_crowdstrike_logs_hmm_training.json    --output-prefix attack_chains to see chains

