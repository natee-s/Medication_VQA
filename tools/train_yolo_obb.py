import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO-OBB drug label detector.")
    parser.add_argument("--data", required=True, help="Path to YOLO OBB dataset yaml.")
    parser.add_argument("--model", default="yolo11n-obb.pt", help="Smallest YOLO OBB base model.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs/drug_label_obb")
    parser.add_argument("--name", default="yolo11n_obb")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    print(result)
    print(f"Best model should be under: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
