# Run your options trading workflow
python complete_async_framework.py --workflow options_trading_workflow.json --concurrent 3 --verbose

# Run your SAP workflow with data
python complete_async_framework.py --workflow sap_better.json --data sap_data.csv --concurrent 5


AND

for older files (final_async_framework(version).txt)
python async_framework_main.py --workflow sap_better.json --data sap_data_test.csv --concurrent 5


SANIC FRONTEND

frontend_complete_asyc_framework.py
AND 
templates/index.html

These are front end framework files to give sanic based front end for the platform


