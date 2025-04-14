#!/bin/bash

# Output file for combined SAP data
output_file="sap_all_combined_data.txt"

# Empty the output file if it exists
> "$output_file"

# List of SAP-related CSV files
files=(
'Batch_characteristic_data_MCH1.csv'
'Inventory_transaction_data_MSEG.csv'
'Batch_classification_data.csv'
'Physical_inventory_data_IKPF_ISEG.csv'
'Pricing_condition_tables_KONP.csv'
'Material_valuation_data_MBEW_v2.csv'
'Material_master_basic_data_MARA_MARC.csv'
'Pricing_schema_data_KONV.csv'
'Purchase_orders_EKKO_EKPO.csv'
'Purchase_requisitions_EBAN.csv'
'Vendor_contact_information_ADR6.csv'
'Vendor_master_data_LFA1.csv'
'Vendor_purchasing_organization_data_LFM1_v2.csv'
'ME2L.csv'
'ME80FN.csv'
'MB51MB51.csv'
'ME21.csv'
'Export_Foreign_Trade_Functionality_VEI_transactions.csv'
'SD_Configuration_transactions.csv'
'Export_Foreign_Trade_Functionality_VEransactions.csv'
'Text_Management_SO10.csv'
'Billing_Plans_and_Periodic_Billing_VA41.csv'
'Rebates_and_Bonuses_VB02.csv'
'Credit_Management_FD32.csv'
'Transportation_Planning_VT01N.csv'
'VT02N_Transportation_Planning.csv'
'Rebates_and_Bonuses_VB01.csv'
'Availability_Check_VA05_Stock.csv'
'VA35 - Sales Returns List.csv'
'Availability_Check_MD04.csv'
'Pricing and Conditions (VK11, VK12, VK13).csv'
'Master Data transactions for materials, customers (MM01).csv'
'MC.9 - Customer Master Data Analysis (1).csv'
'MC.9 - Customer Master Data Analysis.csv'
'VTCB - Backorder Processing.csv'
'VF04 - Billing Document Analysis.csv'
'VA45L - Credit Memo Lists.csv'
'VA15 - Inquiries List.csv'
'S_ALR_87012215 - Sales Information System Report.csv'
'VKM5 - Customer Analysis.csv'
'VT04 - Contracts List.csv'
'VA25 - Quotations List.csv'
'MB51 - Material Document List.csv'
'VF05 - Billing Document List.csv'
'VL06O - Outbound Deliveries.csv'
'VA03 - Display Sales Order.csv'
'Profitability Analysis Reports (S_ALR_87013611).csv'
'Tax Reconciliation Reports (S_ALR_87012357).csv'
'Financial Planning and Budgeting Reports (S_ALR_87013625).csv'
'Treasury Reports (S_ALR_87012215).csv'
'Month-End Close Reports (S_ALR_87012289).csv'
'Revenue Analysis by Customer (S_P99_41000029).csv'
'Budget vs. Actual Comparison (S_ALR_87013542).csv'
'Journal Entry Analysis (S_ALR_87012289).csv'
'Cash Management and Forecasting (FF7AN).csv'
'Intercompany Reconciliation (S_E38_98000088).csv'
'Customer Master Data Report (S_ALR_87012082).csv'
'Trial Balance Report (S_ALR_87012301).csv'
'Vendor Master Data Report (S_ALR_87012086).csv'
'Asset Accounting Reports (S_ALR_87011990).csv'
'Cost Center Reports (S_ALR_87013611).csv'
'Profit_and_Loss_Analysis_S_ALR_87012993.csv'
'Accounts_Receivable_Aging_FBL5N.csv'
'Accounts_Payable_Aging_FBL1N.csv'
'General_Ledger_Account_Balances_FBL3N.csv'
'Financial_Statements_Balance_Income_CashFlow.csv'
)

# Loop and merge contents
for file in "${files[@]}"; do
  echo "File: $file" >> "$output_file"
  cat "$file" >> "$output_file"
  echo -e "\n\n" >> "$output_file"
done

echo "SAP files combined into $output_file."
