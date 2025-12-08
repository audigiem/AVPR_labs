#!/bin/bash

# Quick status check script for LAB5 jobs on ensicompute
# Usage: ./check_status.sh [job_id]

echo "============================================================================"
echo "LAB5 - Job Status Checker"
echo "============================================================================"
echo ""

# Check if running on nash
if [[ $(hostname) != "nash"* ]]; then
    echo "⚠️  This script should be run on nash.ensimag.fr"
    echo "   Connect first: ssh your_login@nash.ensimag.fr"
    exit 1
fi

# If job ID provided, show specific job
if [ ! -z "$1" ]; then
    echo "📊 Status for Job ID: $1"
    echo "────────────────────────────────────────────────────────────────────────"
    squeue -j $1 -o "%.18i %.9P %.30j %.8u %.2t %.10M %.10l %.6D %R"
    echo ""

    # Try to find and display recent output
    latest_out=$(ls -t cluster_logs/output/*.out 2>/dev/null | head -1)
    if [ ! -z "$latest_out" ]; then
        echo "📄 Latest output file: $latest_out"
        echo "────────────────────────────────────────────────────────────────────────"
        echo "Last 20 lines:"
        tail -20 "$latest_out"
    fi
else
    # Show all user's jobs
    echo "📊 Your Jobs:"
    echo "────────────────────────────────────────────────────────────────────────"
    squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.10l %.6D %R"

    job_count=$(squeue -u $USER | wc -l)
    job_count=$((job_count - 1))  # Remove header line

    if [ $job_count -eq 0 ]; then
        echo "No jobs currently running or pending"
    else
        echo ""
        echo "Total: $job_count job(s)"
    fi

    echo ""
    echo "Legend:"
    echo "  PD = Pending (waiting for resources)"
    echo "  R  = Running"
    echo "  CG = Completing"
    echo "  CD = Completed"
    echo "  F  = Failed"
    echo ""
fi

echo "────────────────────────────────────────────────────────────────────────"

# Show recent log files
echo ""
echo "📁 Recent Log Files:"
echo "────────────────────────────────────────────────────────────────────────"

if [ -d "cluster_logs/output" ]; then
    echo "Output logs:"
    ls -lth cluster_logs/output/*.out 2>/dev/null | head -5 | awk '{print "  " $9 " (" $5 ", " $6 " " $7 " " $8 ")"}'
fi

if [ -d "cluster_logs/errors" ]; then
    echo ""
    echo "Error logs:"
    ls -lth cluster_logs/errors/*.err 2>/dev/null | head -5 | awk '{print "  " $9 " (" $5 ", " $6 " " $7 " " $8 ")"}'

    # Check for non-empty error files
    error_files=$(find cluster_logs/errors -name "*.err" -type f -size +0 2>/dev/null)
    if [ ! -z "$error_files" ]; then
        echo ""
        echo "⚠️  Warning: Some error files are not empty:"
        echo "$error_files" | while read file; do
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "  - $file ($size)"
        done
    fi
fi

echo ""
echo "────────────────────────────────────────────────────────────────────────"

# Show GPU usage on cluster
echo ""
echo "🖥️  Cluster GPU Status:"
echo "────────────────────────────────────────────────────────────────────────"
sinfo -o "%20P %5a %.10l %16F %N" | head -10

echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "Commands:"
echo "  ./check_status.sh <job_id>                    # Check specific job"
echo "  tail -f cluster_logs/output/<file>.out        # Follow output"
echo "  tail -f cluster_logs/errors/<file>.err        # Follow errors"
echo "  scancel <job_id>                              # Cancel job"
echo "============================================================================"

