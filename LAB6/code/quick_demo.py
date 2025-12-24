"""
Script de Démonstration et Tests Rapides
Pour tester rapidement les fonctionnalités sans entraînement complet
"""

import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torchvision.models.detection as detection
from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
import os


def demo_pretrained_detection(image_path="family-and-dog.jpg"):
    """
    Démo 1: Détection d'objets avec un modèle pré-entraîné
    Utilise Faster R-CNN pré-entraîné sur COCO
    """
    print("=" * 60)
    print("DEMO 1: Détection avec Modèle Pré-entraîné")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"ERREUR: Image '{image_path}' non trouvée!")
        return

    # Charger le modèle
    print("\n1. Chargement du modèle Faster R-CNN...")
    model = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()

    # Obtenir les classes COCO
    classes = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
    print(f"   Modèle chargé avec {len(classes)} classes COCO")

    # Charger et prétraiter l'image
    print(f"\n2. Chargement de l'image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"   Dimensions: {image.size}")

    transform = transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0)

    # Détection
    print("\n3. Exécution de la détection...")
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]

    print(f"   Détections brutes: {len(boxes)}")

    # Filtrage par confiance
    confidence_threshold = 0.5
    mask = scores >= confidence_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    print(f"   Après filtrage (conf > {confidence_threshold}): {len(filtered_boxes)}")

    # Application du NMS
    if len(filtered_boxes) > 0:
        keep = nms(filtered_boxes, filtered_scores, iou_threshold=0.5)
        final_boxes = filtered_boxes[keep]
        final_scores = filtered_scores[keep]
        final_labels = filtered_labels[keep]

        print(f"   Après NMS: {len(final_boxes)}")

        # Afficher les détections
        print("\n4. Objets détectés:")
        for i, (box, score, label) in enumerate(
            zip(final_boxes, final_scores, final_labels)
        ):
            class_name = classes[label.item()]
            print(f"   [{i+1}] {class_name}: {score.item():.3f} - Box: {box.tolist()}")

        # Visualisation
        print("\n5. Création de la visualisation...")
        image_draw = image.copy()
        draw = ImageDraw.Draw(image_draw)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for box, score, label in zip(final_boxes, final_scores, final_labels):
            x1, y1, x2, y2 = box.tolist()
            class_name = classes[label.item()]

            # Dessiner la boîte
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

            # Dessiner le label
            text = f"{class_name} {score.item():.2f}"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            draw.rectangle([(x1, y1 - text_height), (x1 + text_width, y1)], fill="red")
            draw.text((x1, y1 - text_height), text, fill="white", font=font)

        output_path = "demo_detection_output.jpg"
        image_draw.save(output_path)
        print(f"   Sauvegardé: {output_path}")
    else:
        print("   Aucune détection après filtrage!")


def demo_compare_models(image_path="family-and-dog.jpg"):
    """
    Démo 2: Comparaison de différents modèles de détection
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Comparaison de Modèles")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"ERREUR: Image '{image_path}' non trouvée!")
        return

    # Charger l'image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0)

    models = [
        ("Faster R-CNN ResNet50", detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")),
        ("RetinaNet ResNet50", detection.retinanet_resnet50_fpn(weights="DEFAULT")),
    ]

    results = {}

    for model_name, model in models:
        print(f"\n--- {model_name} ---")
        model.eval()

        import time

        start = time.time()

        with torch.no_grad():
            predictions = model(img_tensor)[0]

        inference_time = time.time() - start

        # Filtrer par confiance
        mask = predictions["scores"] >= 0.5
        num_detections = mask.sum().item()

        results[model_name] = {
            "inference_time": inference_time,
            "num_detections": num_detections,
            "avg_confidence": (
                predictions["scores"][mask].mean().item() if num_detections > 0 else 0
            ),
        }

        print(f"Temps d'inférence: {inference_time:.4f}s")
        print(f"Détections (conf > 0.5): {num_detections}")
        if num_detections > 0:
            print(f"Confiance moyenne: {results[model_name]['avg_confidence']:.3f}")

    print("\n--- Résumé de la Comparaison ---")
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Temps: {result['inference_time']:.4f}s")
        print(f"  Détections: {result['num_detections']}")
        print(f"  Confiance moy.: {result['avg_confidence']:.3f}")


def demo_nms_comparison(image_path="family-and-dog.jpg"):
    """
    Démo 3: Impact du NMS avec différents seuils IoU
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Impact du NMS")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"ERREUR: Image '{image_path}' non trouvée!")
        return

    # Charger modèle et image
    model = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0)

    # Détection
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # Filtrer par confiance
    mask = predictions["scores"] >= 0.5
    boxes = predictions["boxes"][mask]
    scores = predictions["scores"][mask]
    labels = predictions["labels"][mask]

    print(f"\nDétections après filtrage (conf > 0.5): {len(boxes)}")

    # Tester différents seuils IoU
    iou_thresholds = [0.3, 0.5, 0.7, 0.9]

    print("\nImpact des différents seuils IoU du NMS:")
    for iou_threshold in iou_thresholds:
        if len(boxes) > 0:
            keep = nms(boxes, scores, iou_threshold)
            print(f"  IoU = {iou_threshold}: {len(keep)} détections conservées")

            # Créer visualisation pour IoU=0.5
            if iou_threshold == 0.5:
                visualize_boxes(
                    image,
                    boxes[keep],
                    scores[keep],
                    labels[keep],
                    f"demo_nms_iou{iou_threshold}.jpg",
                )


def visualize_boxes(image, boxes, scores, labels, output_path):
    """Fonction utilitaire pour visualiser les boîtes"""
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    classes = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        class_name = (
            classes[label.item()]
            if label.item() < len(classes)
            else f"Class {label.item()}"
        )

        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

        text = f"{class_name} {score.item():.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.rectangle([(x1, y1 - text_height), (x1 + text_width, y1)], fill="red")
        draw.text((x1, y1 - text_height), text, fill="white", font=font)

    image_draw.save(output_path)
    print(f"   Visualisation sauvegardée: {output_path}")


def demo_custom_inference_pipeline():
    """
    Démo 4: Pipeline complet d'inférence personnalisé
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Pipeline d'Inférence Personnalisé")
    print("=" * 60)

    class ObjectDetector:
        def __init__(
            self, model_name="fasterrcnn", confidence_threshold=0.5, iou_threshold=0.5
        ):
            print(f"\nInitialisation du détecteur...")
            print(f"  Modèle: {model_name}")
            print(f"  Seuil de confiance: {confidence_threshold}")
            print(f"  Seuil IoU NMS: {iou_threshold}")

            self.confidence_threshold = confidence_threshold
            self.iou_threshold = iou_threshold

            if model_name == "fasterrcnn":
                self.model = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            elif model_name == "retinanet":
                self.model = detection.retinanet_resnet50_fpn(weights="DEFAULT")
            else:
                raise ValueError(f"Modèle inconnu: {model_name}")

            self.model.eval()
            self.classes = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
            self.transform = transforms.ToTensor()

        def detect(self, image_path):
            """Détecter des objets dans une image"""
            # Charger image
            image = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(image).unsqueeze(0)

            # Inférence
            with torch.no_grad():
                predictions = self.model(img_tensor)[0]

            # Filtrage
            mask = predictions["scores"] >= self.confidence_threshold
            boxes = predictions["boxes"][mask]
            scores = predictions["scores"][mask]
            labels = predictions["labels"][mask]

            # NMS
            if len(boxes) > 0:
                keep = nms(boxes, scores, self.iou_threshold)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            # Formater résultats
            results = []
            for box, score, label in zip(boxes, scores, labels):
                results.append(
                    {
                        "class": self.classes[label.item()],
                        "confidence": score.item(),
                        "bbox": box.tolist(),
                    }
                )

            return results, image

        def visualize(self, image, results, output_path):
            """Visualiser les résultats"""
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            for result in results:
                x1, y1, x2, y2 = result["bbox"]
                draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)

                text = f"{result['class']} {result['confidence']:.2f}"
                draw.text((x1, y1 - 20), text, fill="green", font=font)

            image.save(output_path)
            print(f"   Résultat sauvegardé: {output_path}")

    # Utiliser le pipeline
    image_path = "family-and-dog.jpg"
    if os.path.exists(image_path):
        detector = ObjectDetector(
            model_name="fasterrcnn", confidence_threshold=0.6, iou_threshold=0.5
        )

        print(f"\nDétection sur: {image_path}")
        results, image = detector.detect(image_path)

        print(f"\nObjets détectés: {len(results)}")
        for i, result in enumerate(results):
            print(f"  [{i+1}] {result['class']}: {result['confidence']:.3f}")

        detector.visualize(image, results, "demo_pipeline_output.jpg")
    else:
        print(f"Image '{image_path}' non trouvée!")


def main():
    """Menu principal des démos"""
    print("=" * 60)
    print("SCRIPTS DE DÉMONSTRATION - LAB6")
    print("Tests Rapides Sans Entraînement")
    print("=" * 60)

    print("\nDémos disponibles:")
    print("1. Détection avec modèle pré-entraîné")
    print("2. Comparaison de modèles")
    print("3. Impact du NMS")
    print("4. Pipeline d'inférence personnalisé")
    print("5. Exécuter toutes les démos")
    print("0. Quitter")

    choice = input("\nChoisir une démo (0-5): ").strip()

    if choice == "1":
        demo_pretrained_detection()
    elif choice == "2":
        demo_compare_models()
    elif choice == "3":
        demo_nms_comparison()
    elif choice == "4":
        demo_custom_inference_pipeline()
    elif choice == "5":
        demo_pretrained_detection()
        demo_compare_models()
        demo_nms_comparison()
        demo_custom_inference_pipeline()
        print("\n" + "=" * 60)
        print("TOUTES LES DÉMOS TERMINÉES!")
        print("=" * 60)
    elif choice == "0":
        print("Au revoir!")
    else:
        print("Choix invalide!")


if __name__ == "__main__":
    main()
