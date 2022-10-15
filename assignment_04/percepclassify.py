import sys


def main(model_file_path: str, input_file_path: str):
    print(model_file_path, input_file_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python <model_file> <input_file>")
        exit(1)

    model_file_path = sys.argv[1]
    input_file_path = sys.argv[2]

    main(model_file_path, input_file_path)
