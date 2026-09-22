# Update safety

The release changes only its own extension folder. It does not patch Forge's
disk files, upgrade the environment or write to other extensions. Runtime hooks
still share Forge APIs and cannot guarantee future compatibility.

## Before updating

1. Stop Forge completely.
2. Save the known-working extension ZIP outside Forge. Back up models and outputs
   separately; never include personal data in public uploads.
3. Record the working Forge commit and environment. The extension ZIP does not
   back up Forge, drivers or shared Python packages.
4. From the extension folder, run:

   ```text
   ../../venv/Scripts/python.exe verify_integrity.py
   ```

The release manifest records SHA-256 hashes of shipped source files. Verification
reports changed, missing and unexpected source files. It detects accidental
changes; it is not a signature or protection against a rewritten manifest.

## After updating

Run verification again. Restart Forge, refresh the browser, and test a small
txt2img image, img2img, and switching checkpoints. Never update during inference.

If integrity changed unexpectedly, stop Forge and restore your trusted extension
backup. If integrity passes but Forge compatibility broke, disable the extension
and restart. Use a compatible extension release or your separately backed-up
Forge environment. Do not overwrite unrelated user changes.

No read-only flag or manifest can guarantee every updater leaves files alone.
This check does not freeze or automatically roll back Forge, drivers or packages.
