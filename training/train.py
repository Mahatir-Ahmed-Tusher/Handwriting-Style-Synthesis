import tensorflow as tf
from tensorflow.keras import layers, optimizers
from models.style_encoder import StyleEncoder
from models.text_generator import TextGenerator
from models.char_discriminator import CharacterDiscriminator
from models.cursive_discriminator import CursiveDiscriminator
from datasets.load_data import load_dataset
from utils.losses import discriminator_loss, generator_loss
import numpy as np
import os

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE_G = 0.0001
LEARNING_RATE_D = 0.0001
LATENT_DIM = 128
IMAGE_SIZE = (128, 128, 1)

# Initialize models
style_encoder = StyleEncoder()
text_generator = TextGenerator()
char_discriminator = CharacterDiscriminator()
cursive_discriminator = CursiveDiscriminator()

# Optimizers
optimizer_G = optimizers.Adam(learning_rate=LEARNING_RATE_G, beta_1=0.5)
optimizer_D = optimizers.Adam(learning_rate=LEARNING_RATE_D, beta_1=0.5)

# Load dataset
train_dataset = load_dataset(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

# Checkpoint directory
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    optimizer_G=optimizer_G,
    optimizer_D=optimizer_D,
    style_encoder=style_encoder,
    text_generator=text_generator,
    char_discriminator=char_discriminator,
    cursive_discriminator=cursive_discriminator
)

# Training step
@tf.function
def train_step(printed_images, real_images, writer_ids):
    with tf.GradientTape(persistent=True) as tape:
        # Generate style vectors
        style_vectors = style_encoder(real_images)

        # Generate synthetic images
        synthetic_images = text_generator(style_vectors, printed_images)

        # Discriminator outputs
        real_char_output = char_discriminator(real_images)
        synthetic_char_output = char_discriminator(synthetic_images)
        real_cursive_output = cursive_discriminator(real_images)
        synthetic_cursive_output = cursive_discriminator(synthetic_images)

        # Discriminator losses
        char_loss = discriminator_loss(real_char_output, synthetic_char_output)
        cursive_loss = discriminator_loss(real_cursive_output, synthetic_cursive_output)
        total_discriminator_loss = char_loss + cursive_loss

        # Generator loss
        total_generator_loss = generator_loss(synthetic_char_output, synthetic_cursive_output)

    # Apply gradients for discriminators
    gradients_D = tape.gradient(total_discriminator_loss, char_discriminator.trainable_variables + cursive_discriminator.trainable_variables)
    optimizer_D.apply_gradients(zip(gradients_D, char_discriminator.trainable_variables + cursive_discriminator.trainable_variables))

    # Apply gradients for generator and style encoder
    gradients_G = tape.gradient(total_generator_loss, text_generator.trainable_variables + style_encoder.trainable_variables)
    optimizer_G.apply_gradients(zip(gradients_G, text_generator.trainable_variables + style_encoder.trainable_variables))

    return total_discriminator_loss, total_generator_loss

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in dataset:
            printed_images, real_images, writer_ids = batch
            d_loss, g_loss = train_step(printed_images, real_images, writer_ids)

            # Print losses
            print(f"Discriminator Loss: {d_loss.numpy()}, Generator Loss: {g_loss.numpy()}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

# Start training
train(train_dataset, EPOCHS)