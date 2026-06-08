package com.example.semanticsegmentation;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.example.semanticsegmentation.databinding.ActivityMainBinding;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * MainActivity — Semantic Segmentation App
 *
 * Pipeline:
 *  1. User selects an image from storage/camera roll
 *  2. Image is pre-processed (resize 128×128, normalize ImageNet stats)
 *  3. TorchScript model (unet_mobile.ptl) runs FP32 inference
 *  4. Argmax over 3 class channels produces pixel-wise mask
 *  5. Mask is colorized (Red=Pet, Green=Background, Blue=Border)
 *  6. Original image, mask, and semi-transparent overlay are displayed
 *  7. Inference latency is shown in the UI
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "SegmentationApp";

    // Model expects 128×128 input (must match training IMAGE_SIZE)
    private static final int MODEL_INPUT_SIZE = 128;
    private static final int NUM_CLASSES = 3;

    // ImageNet normalization — same values used in training transforms
    private static final float[] NORM_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] NORM_STD  = {0.229f, 0.224f, 0.225f};

    // Class colours — Pet=Red, Background=Green, Border=Blue
    private static final int[] CLASS_COLORS = {
            Color.argb(200, 255, 0,   0),   // 0 — Pet
            Color.argb(200, 0,   255, 0),   // 1 — Background
            Color.argb(200, 0,   0,   255)  // 2 — Border
    };

    private ActivityMainBinding binding;
    private Module torchModule;
    private Bitmap selectedBitmap;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    // ------------------------------------------------------------------ //
    //  Activity lifecycle
    // ------------------------------------------------------------------ //

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        loadModel();
        setupButtons();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdown();
        if (torchModule != null) torchModule.destroy();
    }

    // ------------------------------------------------------------------ //
    //  Model loading
    // ------------------------------------------------------------------ //

    /**
     * Copies model from assets to internal files dir (PyTorch requires a real path)
     * and loads it as a TorchScript Module.
     */
    private void loadModel() {
        executor.execute(() -> {
            try {
                String modelPath = assetFilePath("unet_mobile.ptl");
                torchModule = Module.load(modelPath);
                Log.i(TAG, "Model loaded successfully from: " + modelPath);
                runOnUiThread(() ->
                        Toast.makeText(this, "Model loaded ✓", Toast.LENGTH_SHORT).show());
            } catch (Exception e) {
                Log.e(TAG, "Failed to load model", e);
                runOnUiThread(() ->
                        Toast.makeText(this,
                                "Model not found — place unet_mobile.ptl in app/src/main/assets/",
                                Toast.LENGTH_LONG).show());
            }
        });
    }

    /**
     * Copies an asset file to the app's internal storage and returns its path.
     * PyTorch's Module.load() requires a filesystem path, not an InputStream.
     */
    private String assetFilePath(String assetName) throws IOException {
        File file = new File(getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) return file.getAbsolutePath();

        try (InputStream is = getAssets().open(assetName);
             OutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[4096];
            int read;
            while ((read = is.read(buffer)) != -1) os.write(buffer, 0, read);
        }
        return file.getAbsolutePath();
    }

    // ------------------------------------------------------------------ //
    //  UI / button wiring
    // ------------------------------------------------------------------ //

    private void setupButtons() {
        binding.btnSelectImage.setOnClickListener(v -> checkPermissionAndPick());
        binding.btnRunInference.setOnClickListener(v -> runInference());
    }

    // ------------------------------------------------------------------ //
    //  Permission & image picker
    // ------------------------------------------------------------------ //

    private final ActivityResultLauncher<String> permissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) openImagePicker();
                else Toast.makeText(this, "Storage permission required", Toast.LENGTH_SHORT).show();
            });

    private final ActivityResultLauncher<Intent> imagePickerLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri uri = result.getData().getData();
                    if (uri != null) handleSelectedImage(uri);
                }
            });

    private void checkPermissionAndPick() {
        String permission = Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
                ? Manifest.permission.READ_MEDIA_IMAGES
                : Manifest.permission.READ_EXTERNAL_STORAGE;

        if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED) {
            openImagePicker();
        } else {
            permissionLauncher.launch(permission);
        }
    }

    private void openImagePicker() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");
        imagePickerLauncher.launch(intent);
    }

    private void handleSelectedImage(Uri uri) {
        try {
            Bitmap bmp = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
            selectedBitmap = bmp;
            binding.ivOriginal.setImageBitmap(bmp);
            binding.ivOverlayBase.setImageBitmap(bmp);
            binding.ivMask.setImageBitmap(null);
            binding.ivOverlayMask.setVisibility(View.GONE);
            binding.tvOverlayHint.setVisibility(View.VISIBLE);
            binding.tvMetrics.setText("");
            binding.btnRunInference.setEnabled(torchModule != null);
        } catch (IOException e) {
            Log.e(TAG, "Failed to load image", e);
            Toast.makeText(this, "Could not load image", Toast.LENGTH_SHORT).show();
        }
    }

    // ------------------------------------------------------------------ //
    //  Inference
    // ------------------------------------------------------------------ //

    private void runInference() {
        if (selectedBitmap == null || torchModule == null) return;

        binding.progressBar.setVisibility(View.VISIBLE);
        binding.btnRunInference.setEnabled(false);
        binding.btnSelectImage.setEnabled(false);

        executor.execute(() -> {
            try {
                // 1. Pre-process: resize to 128×128
                Bitmap resized = Bitmap.createScaledBitmap(
                        selectedBitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true);

                // 2. Convert bitmap to normalised float tensor [1, 3, 128, 128]
                Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                        resized, NORM_MEAN, NORM_STD);

                // 3. Forward pass — FP32
                long startMs = System.currentTimeMillis();
                IValue output = torchModule.forward(IValue.from(inputTensor));
                long inferenceMs = System.currentTimeMillis() - startMs;

                // 4. Extract logits tensor [1, 3, 128, 128]
                Tensor outputTensor = output.toTensor();
                float[] scores = outputTensor.getDataAsFloatArray();

                // 5. Argmax over class dimension → pixel labels
                int[] labels = argmax(scores, NUM_CLASSES, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

                // 6. Build coloured mask bitmap
                Bitmap maskBitmap = labelsToBitmap(labels, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

                // 7. Scale mask back to display size of original
                int w = selectedBitmap.getWidth();
                int h = selectedBitmap.getHeight();
                Bitmap maskScaled = Bitmap.createScaledBitmap(maskBitmap, w, h, false);

                final long ms = inferenceMs;
                runOnUiThread(() -> displayResults(maskScaled, ms));

            } catch (Exception e) {
                Log.e(TAG, "Inference error", e);
                runOnUiThread(() ->
                        Toast.makeText(this, "Inference failed: " + e.getMessage(),
                                Toast.LENGTH_LONG).show());
            } finally {
                runOnUiThread(() -> {
                    binding.progressBar.setVisibility(View.GONE);
                    binding.btnRunInference.setEnabled(true);
                    binding.btnSelectImage.setEnabled(true);
                });
            }
        });
    }

    // ------------------------------------------------------------------ //
    //  Post-processing helpers
    // ------------------------------------------------------------------ //

    /**
     * Computes argmax over the class dimension of the output tensor.
     *
     * The tensor layout coming from PyTorch is [N, C, H, W] stored as flat float[].
     * For N=1: index = c * H * W + h * W + w
     *
     * @param scores     flat float array from the output tensor
     * @param numClasses number of segmentation classes (3)
     * @param height     spatial height (128)
     * @param width      spatial width (128)
     * @return int[] of length H*W, each value in [0, numClasses)
     */
    private int[] argmax(float[] scores, int numClasses, int height, int width) {
        int pixelCount = height * width;
        int[] labels = new int[pixelCount];

        for (int pixel = 0; pixel < pixelCount; pixel++) {
            float maxScore = Float.NEGATIVE_INFINITY;
            int bestClass = 0;
            for (int c = 0; c < numClasses; c++) {
                float s = scores[c * pixelCount + pixel];
                if (s > maxScore) {
                    maxScore = s;
                    bestClass = c;
                }
            }
            labels[pixel] = bestClass;
        }
        return labels;
    }

    /**
     * Converts integer label map → ARGB Bitmap using per-class colours.
     */
    private Bitmap labelsToBitmap(int[] labels, int width, int height) {
        int[] pixels = new int[width * height];
        for (int i = 0; i < labels.length; i++) {
            pixels[i] = CLASS_COLORS[labels[i]];
        }
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bmp.setPixels(pixels, 0, width, 0, 0, width, height);
        return bmp;
    }

    // ------------------------------------------------------------------ //
    //  Display
    // ------------------------------------------------------------------ //

    private void displayResults(Bitmap maskBitmap, long inferenceMs) {
        // Panel 2: mask alone
        binding.ivMask.setImageBitmap(maskBitmap);

        // Panel 3: overlay (mask at 50% alpha on top of original)
        binding.ivOverlayMask.setImageBitmap(maskBitmap);
        binding.ivOverlayMask.setVisibility(View.VISIBLE);
        binding.tvOverlayHint.setVisibility(View.GONE);

        // Metrics
        binding.tvMetrics.setText(String.format("Inference: %d ms  |  Size: %d×%d  |  Classes: Pet / Background / Border",
                inferenceMs, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE));

        Toast.makeText(this, "Segmentation complete ✓", Toast.LENGTH_SHORT).show();
    }
}
