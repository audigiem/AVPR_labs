"""
LAB5 - Complete Task Runner
Execute all tasks or individual tasks for the LAB5 exercises.
"""

import sys
import argparse

def print_menu():
    """Display the main menu"""
    print("\n" + "="*70)
    print("LAB5 - IMAGE RECOGNITION USING DEEP CONVOLUTIONAL NEURAL NETWORKS")
    print("="*70)
    print("\nAvailable Tasks:")
    print("  1. Task 1: Hyperparameter Exploration")
    print("     - Learning Rate Adjustment")
    print("     - Batch Size Variance")
    print("     - Epoch Sensitivity")
    print("\n  2. Task 2: Architectural Adaptation")
    print("     - Layer Modification (Dropout)")
    print("     - Filter Size Adjustments")
    print("     - Feature Map Depth Variation")
    print("\n  3. Task 3: Data Transformation Techniques")
    print("     - Transformation Sequence Experimentation")
    print("     - Data Augmentation Trials")
    print("     - Normalization and Standardization")
    print("\n  4. Run All Tasks")
    print("  5. Exit")
    print("="*70)


def run_task1():
    """Run Task 1: Hyperparameter Exploration"""
    print("\nRunning Task 1: Hyperparameter Exploration...")
    import task1_hyperparameter_exploration
    task1_hyperparameter_exploration.main()


def run_task2():
    """Run Task 2: Architectural Adaptation"""
    print("\nRunning Task 2: Architectural Adaptation...")
    import task2_architectural_adaptation
    task2_architectural_adaptation.main()


def run_task3():
    """Run Task 3: Data Transformation Techniques"""
    print("\nRunning Task 3: Data Transformation Techniques...")
    import task3_data_transformation
    task3_data_transformation.main()


def run_all_tasks():
    """Run all tasks sequentially"""
    print("\n" + "="*70)
    print("RUNNING ALL TASKS SEQUENTIALLY")
    print("="*70)

    try:
        run_task1()
    except Exception as e:
        print(f"\nError in Task 1: {e}")
        print("Continuing to next task...\n")

    try:
        run_task2()
    except Exception as e:
        print(f"\nError in Task 2: {e}")
        print("Continuing to next task...\n")

    try:
        run_task3()
    except Exception as e:
        print(f"\nError in Task 3: {e}")

    print("\n" + "="*70)
    print("ALL TASKS COMPLETED!")
    print("="*70)


def main():
    """Main function with CLI support"""
    parser = argparse.ArgumentParser(
        description='LAB5 - Image Recognition Tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lab5_runner.py              # Interactive menu
  python lab5_runner.py --task 1     # Run Task 1 only
  python lab5_runner.py --task all   # Run all tasks
        """
    )

    parser.add_argument('--task', type=str, choices=['1', '2', '3', 'all'],
                       help='Task number to run (1, 2, 3, or all)')

    args = parser.parse_args()

    # If task argument is provided, run directly
    if args.task:
        if args.task == '1':
            run_task1()
        elif args.task == '2':
            run_task2()
        elif args.task == '3':
            run_task3()
        elif args.task == 'all':
            run_all_tasks()
        return

    # Interactive menu mode
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == '1':
            run_task1()
            input("\nPress Enter to continue...")
        elif choice == '2':
            run_task2()
            input("\nPress Enter to continue...")
        elif choice == '3':
            run_task3()
            input("\nPress Enter to continue...")
        elif choice == '4':
            run_all_tasks()
            input("\nPress Enter to continue...")
        elif choice == '5':
            print("\nExiting... Goodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice! Please enter a number between 1 and 5.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()

