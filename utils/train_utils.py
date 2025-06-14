import torch
import os
import wandb
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def denormalize_log_prediction(log_depth):

    # Convert from log space back to linear depth in meters
    depth_map = torch.exp(log_depth)

    return depth_map


def train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs, device, exp_path,
                mask_indicator=None, log_input=False, el_loss=False):
    """Train the model and save the best based on validation metrics"""
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    if mask_indicator is not None and log_input:
        # We simply want the mask indicator to be out of bounds for any adequate data points we have
        mask_indicator = -1e20
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, targets, _ in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            if mask_indicator:
                with torch.no_grad():
                    mask = (targets != mask_indicator)
                    masked_targets = targets * mask

            # Forward pass
            outputs = model(inputs)

            if mask_indicator:
                outputs = outputs * mask
                if el_loss:
                    inputs_orig = nn.functional.interpolate(
                            inputs,
                            size=(426, 560),  # Original input dimensions
                            mode='bilinear',
                            align_corners=True
                    )
                    loss = (mask.numel()/max(mask.numel()//2, mask.sum())) * loss_func(inputs_orig,
                                                                                       outputs,
                                                                                       masked_targets)
                else:
                    loss = (mask.numel()/max(mask.numel()//2, mask.sum())) * loss_func(outputs,
                                                                                       masked_targets)
                # print(mask.sum(), mask.numel(), torch.min(masked_targets))
                # print(loss)
            else:
                if el_loss:
                    inputs_orig = nn.functional.interpolate(
                            inputs,
                            size=(426, 560),  # Original input dimensions
                            mode='bilinear',
                            align_corners=True
                    )
                    loss = loss_func(inputs_orig, outputs, targets)
                else:
                    loss = loss_func(outputs, targets)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                if mask_indicator:
                    mask = (targets != mask_indicator)
                    masked_targets = targets * mask
                # Forward pass
                outputs = model(inputs)

                if mask_indicator:
                    outputs = outputs * mask
                    if el_loss:
                        inputs_orig = nn.functional.interpolate(
                            inputs,
                            size=(426, 560),  # Original input dimensions
                            mode='bilinear',
                            align_corners=True
                        )
                        loss = loss_func(inputs_orig, outputs, masked_targets)
                    else:
                        loss = loss_func(outputs, masked_targets)
                else:
                    if el_loss:
                        inputs_orig = nn.functional.interpolate(
                            inputs,
                            size=(426, 560),  # Original input dimensions
                            mode='bilinear',
                            align_corners=True
                        )
                        loss = loss_func(inputs_orig, outputs, targets)
                    else:
                        loss = loss_func(outputs, targets)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        ### I guess I need to add it into the previous loop, but for now it will do ####
        ### Additional metrics logging ####
        evaluate_model(model, val_loader, device, exp_path=None, epoch=epoch,
                       mask_indicator=mask_indicator, log_input=log_input)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        ### LOSS WANDB LOGGING ###

        wandb.log({"train/train": train_loss}, epoch)
        wandb.log({"val/val": val_loss}, epoch)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
            torch.save(model.state_dict(), f'{exp_path}/best_model_{epoch}.pt')
            print(
                f"New best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")

    print(
        f"\nBest model was from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")

    # Load the best model
    model.load_state_dict(torch.load(f'{exp_path}/best_model_{best_epoch}.pt'))

    return model


def evaluate_model(model, val_loader, device, exp_path, epoch=None,
                   mask_indicator=None, log_input=False):
    """Evaluate the model and compute metrics on validation set"""
    model.eval()

    mae = 0.0
    rmse = 0.0
    rel = 0.0
    delta1 = 0.0
    delta2 = 0.0
    delta3 = 0.0
    sirmse = 0.0

    total_samples = 0
    target_shape = None

    valid_pixels = 0
    with torch.no_grad():
        for inputs, targets, filenames in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            if mask_indicator:
                mask = (targets != mask_indicator)
                targets = targets * mask
                valid_pixels += mask.sum().item()

            if target_shape is None:
                target_shape = targets.shape

            # Forward pass

            outputs = model(inputs)

            # Resize outputs to match target dimensions
            outputs = nn.functional.interpolate(
                outputs,
                size=targets.shape[-2:],  # Match height and width of targets
                mode='bilinear',
                align_corners=True
            )

            if mask_indicator:
                outputs = outputs * mask

            if log_input:
                outputs = denormalize_log_prediction(outputs)
                targets = denormalize_log_prediction(targets)

            # Calculate metrics
            abs_diff = torch.abs(outputs - targets)
            mae += torch.sum(abs_diff).item()
            rmse += torch.sum(torch.pow(abs_diff, 2)).item()
            rel += torch.sum(abs_diff / (targets + 1e-6)).item()

            # Calculate scale-invariant RMSE for each image in the batch
            for i in range(batch_size):
                # Convert tensors to numpy arrays
                pred_np = outputs[i].cpu().squeeze().numpy()
                target_np = targets[i].cpu().squeeze().numpy()

                EPSILON = 1e-6

                valid_target = target_np > EPSILON
                if not np.any(valid_target):
                    continue

                target_valid = target_np[valid_target]
                pred_valid = pred_np[valid_target]

                log_target = np.log(target_valid)

                pred_valid = np.where(
                    pred_valid > EPSILON, pred_valid, EPSILON)
                log_pred = np.log(pred_valid)

                # Calculate scale-invariant error
                diff = log_pred - log_target
                diff_mean = np.mean(diff)

                # Calculate RMSE for this image
                sirmse += np.sqrt(np.mean((diff - diff_mean) ** 2))

            # Calculate thresholded accuracy
            max_ratio = torch.max(outputs / (targets + 1e-6),
                                  targets / (outputs + 1e-6))
            delta1 += torch.sum(max_ratio < 1.25).item()
            delta2 += torch.sum(max_ratio < 1.25**2).item()
            delta3 += torch.sum(max_ratio < 1.25**3).item()

            if epoch is None:
                # Save some sample predictions
                if total_samples <= 5 * batch_size:
                    for i in range(min(batch_size, 5)):
                        idx = total_samples - batch_size + i

                        # Convert tensors to numpy arrays
                        input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
                        target_np = targets[i].cpu().squeeze().numpy()
                        output_np = outputs[i].cpu().squeeze().numpy()

                        # Normalize for visualization
                        input_np = (input_np - input_np.min()) / \
                            (input_np.max() - input_np.min() + 1e-6)

                        # Create visualization
                        plt.figure(figsize=(15, 5))

                        plt.subplot(1, 3, 1)
                        plt.imshow(input_np)
                        plt.title("RGB Input")
                        plt.axis('off')

                        plt.subplot(1, 3, 2)
                        plt.imshow(target_np, cmap='plasma')
                        plt.title("Ground Truth Depth")
                        plt.axis('off')

                        plt.subplot(1, 3, 3)
                        plt.imshow(output_np, cmap='plasma')
                        plt.title("Predicted Depth")
                        plt.axis('off')

                        plt.tight_layout()
                        plt.savefig(os.path.join(
                            exp_path, f"sample_{idx}.png"))
                        plt.close()

                # Free up memory
                del inputs, targets, outputs, abs_diff, max_ratio
        if epoch is None:
            # Clear CUDA cache
            torch.cuda.empty_cache()

    # Calculate final metrics using stored target shape

    if mask_indicator:

        mae /= valid_pixels
        rmse = np.sqrt(rmse / valid_pixels)
        rel /= valid_pixels
        sirmse = sirmse / total_samples
        delta1 /= valid_pixels
        delta2 /= valid_pixels
        delta3 /= valid_pixels

    else:
        # channels * height * width
        total_pixels = target_shape[1] * target_shape[2] * target_shape[3]
        mae /= total_samples * total_pixels
        rmse = np.sqrt(rmse / (total_samples * total_pixels))
        rel /= total_samples * total_pixels
        sirmse = sirmse / total_samples
        delta1 /= total_samples * total_pixels
        delta2 /= total_samples * total_pixels
        delta3 /= total_samples * total_pixels

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'siRMSE': sirmse,
        'REL': rel,
        'Delta1': delta1,
        'Delta2': delta2,
        'Delta3': delta3
    }
    if epoch is not None:
        for key in metrics.keys():
            wandb.log({"val/" + str(key): metrics[key]}, epoch)

    return metrics


def generate_test_predictions(model, test_loader, device, exp_path, log_input=False):
    """Generate predictions for the test set without ground truth"""
    model.eval()

    # # Ensure predictions directory exists
    # ensure_dir(predictions_dir)
    os.makedirs(os.path.join(exp_path, "results"), exist_ok=True)
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Generating Test Predictions"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Forward pass
            outputs = model(inputs)

            # Resize outputs to match original input dimensions (426x560)
            outputs = nn.functional.interpolate(
                outputs,
                size=(426, 560),  # Original input dimensions
                mode='bilinear',
                align_corners=True
            )

            if log_input:
                outputs = denormalize_log_prediction(outputs)

            # Save all test predictions
            for i in range(batch_size):
                # Get filename without extension
                filename = filenames[i].split(' ')[1]

                # Save depth map prediction as numpy array
                depth_pred = outputs[i].cpu().squeeze().numpy()
                np.save(os.path.join(os.path.join(
                    exp_path, "results"), f"{filename}"), depth_pred)

            # Clean up memory
            del inputs, outputs

        # Clear cache after test predictions
        torch.cuda.empty_cache()


def visualize_test_predictions(model, test_loader, device, exp_path, log_input=True):
    """Evaluate the model and compute metrics on validation set"""
    model.eval()
    os.makedirs(os.path.join(exp_path, "result_viz"), exist_ok=True)
    cnt = 0
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Visualizing"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            # Forward pass
            outputs = model(inputs)
            # Resize outputs to match target dimensions
            outputs = nn.functional.interpolate(
                outputs,
                size=(426, 560),  # Original input dimensions
                mode='bilinear',
                align_corners=True
            )

            # Resize inputs simply for visualization purposes
            inputs = inputs[...,67:-67, :] # Remove padding
            inputs = nn.functional.interpolate(
                inputs,
                size=(426, 560),  # Original input dimensions
                mode='bilinear',
                align_corners=True
            )

            
            if log_input:
                outputs = denormalize_log_prediction(outputs)
            for i in range(len(inputs)):
                # Convert tensors to numpy arrays
                input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
                output_np = outputs[i].cpu().squeeze().numpy()
                # Normalize for visualization
                input_np = (input_np - input_np.min()) / \
                    (input_np.max() - input_np.min() + 1e-6)
                
                # Create visualization with colorbars
                fig, axes = plt.subplots(1, 2, figsize=(18, 6))
                
                # RGB Input
                axes[0].imshow(input_np)
                axes[0].set_title("RGB Input", fontsize=14)
                axes[0].axis('off')
                
                # Predicted Depth with colorbar
                im = axes[1].imshow(output_np, cmap='plasma')
                axes[1].set_title("Predicted Depth (DPT)", fontsize=14)
                axes[1].axis('off')
                
                # Add colorbar for predicted depth
                cbar = plt.colorbar(im, ax=axes[1], shrink=0.8, aspect=20)
                cbar.set_label('Depth (meters)', rotation=270, labelpad=20, fontsize=12)
                
                
                plt.tight_layout()
                plt.savefig(os.path.join(os.path.join(
                    exp_path, "result_viz"), f"sample_{cnt}.png"), 
                    dpi=150, bbox_inches='tight')
                plt.close()
                cnt += 1
