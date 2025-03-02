import argparse

import formatting


def main():
    parser = argparse.ArgumentParser(description="CNN for predicting numbers")
    parser.add_argument("image_path", help="Path to the input image file")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Run training",
    )

    group.add_argument(
        "-p",
        "--predict",
        type=str,
        metavar="INPUT_STRING",
        help="Run prediction",
    )

    args = parser.parse_args()

    if args.train:
        pass
    elif args.predict:
        img_list = formatting.extract_digits(args.image_path, show_steps=True)


if __name__ == "__main__":
    main()
