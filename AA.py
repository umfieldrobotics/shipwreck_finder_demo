BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = 1
NUM_EPOCHS = 10
LEARNING_RATE = 5e-4
IGNORE_INDEX = -1
TARGET_DIR = Path("/content/segmentation_vis")  # where images/preds/labels are saved
TARGET_DIR.mkdir(parents=True, exist_ok=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE_TEST,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

weight0, weight1 = compute_balanced_weights(train_loader, ignore_index=IGNORE_INDEX)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
weight_tensor = torch.tensor([weight0, weight1], dtype=torch.float32, device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=IGNORE_INDEX)

epoch_bar = tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs")
for epoch_idx in epoch_bar:
    if epoch_idx == 0:
        model.eval()
        running_test_loss = 0.0
        running_test_iou = 0.0
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(DEVICE, non_blocking=True)
                labels = batch["label"].to(DEVICE, non_blocking=True)
                if labels.dim() == 4:
                    labels = labels.squeeze(1)
                logits = model(images)
                loss = criterion(logits, labels)
                running_test_loss += loss.item() * images.size(0)
                running_test_iou += foreground_iou_from_logits(
                    logits, labels, ignore_index=IGNORE_INDEX
                ) * images.size(0)

        mean_test_loss = running_test_loss / len(test_loader.dataset)
        mean_test_iou = running_test_iou / len(test_loader.dataset)
    
    model.train()
    running_train_loss = 0.0
    for batch in train_loader:
        images = batch["image"].to(DEVICE, non_blocking=True)  # [B,3,H,W]
        labels = batch["label"].to(DEVICE, non_blocking=True)  # [B,H,W] or [B,1,H,W]
        if labels.dim() == 4:
            labels = labels.squeeze(1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)  # [B,2,H,W]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * images.size(0)

    mean_train_loss = running_train_loss / len(train_loader.dataset)

    model.eval()
    running_test_loss = 0.0
    running_test_iou = 0.0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            if labels.dim() == 4:
                labels = labels.squeeze(1)
            logits = model(images)
            loss = criterion(logits, labels)
            running_test_loss += loss.item() * images.size(0)
            running_test_iou += foreground_iou_from_logits(
                logits, labels, ignore_index=IGNORE_INDEX
            ) * images.size(0)

    mean_test_loss = running_test_loss / len(test_loader.dataset)
    mean_test_iou = running_test_iou / len(test_loader.dataset)

    epoch_bar.set_postfix(
        {
            "train_loss": f"{mean_train_loss:.4f}",
            "test_loss": f"{mean_test_loss:.4f}",
            "fg_IoU": f"{mean_test_iou:.4f}",
        }
    )


visualize_triplets_inline(model, test_loader, DEVICE, num_items=3)
dump_test_visuals(test_loader, TARGET_DIR, max_items=None)
print(f"Saved test-set visuals to: {TARGET_DIR.resolve()}")
