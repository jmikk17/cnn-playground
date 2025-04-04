import argparse

import cv2

import auxil
import formatting
import train


def main() -> None:
    """Train or run the CNN model.

    Todo:
        * Add a more formal test set of real life imgs to evaluate the model

    """
    parser = argparse.ArgumentParser(description="CNN for predicting numbers")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Run training of CNN model",
    )

    group.add_argument(
        "-p",
        "--predict",
        type=str,
        metavar="INPUT_STRING",
        help="Run prediction of image file",
    )

    args = parser.parse_args()

    if args.train:
        train.run_training()
    elif args.predict:
        device = auxil.get_device()
        img_list, pos, result_img = formatting.extract_digits(args.predict, show_steps=True)
        model = auxil.load_model_for_prediction()
        for i, img in enumerate(img_list):
            pred = auxil.predict_digit(model, device, img)
            cv2.putText(result_img, str(pred), pos[i], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
