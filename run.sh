#!/bin/bash

# Black Box Challenge - Your Implementation
# This script should take three parameters and output the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

DAYS=$1
MILES=$2
RECEIPTS=$3

# TODO: Replace this with your actual implementation
REIMBURSEMENT=$(python3 calculate_reimbursement.py "$DAYS" "$MILES" "$RECEIPTS")
echo "\$${REIMBURSEMENT}"
