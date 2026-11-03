/**
 * Serverless hosting rejects request bodies over about 4.5 MB. Phone photos are often larger,
 * so shrink them in the browser first. Files under the limit are sent untouched.
 */

const MAX_UPLOAD_BYTES = 3.5 * 1024 * 1024;
const MAX_SIDE = 2600;

export async function shrinkForUpload(
  blob: Blob,
  filename: string
): Promise<{ blob: Blob; filename: string }> {
  if (blob.size <= MAX_UPLOAD_BYTES || !blob.type.startsWith('image/')) {
    return { blob, filename };
  }

  const bitmap = await createImageBitmap(blob);
  let scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  let quality = 0.88;
  let result: Blob | null = null;

  // Step down quality, then size, until the file fits.
  for (let attempt = 0; attempt < 6; attempt++) {
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    canvas.getContext('2d')!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    result = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
    if (result && result.size <= MAX_UPLOAD_BYTES) break;
    quality = Math.max(0.6, quality - 0.1);
    scale *= 0.8;
  }
  bitmap.close();

  if (!result) return { blob, filename };
  return { blob: result, filename: filename.replace(/\.[^.]+$/, '') + '.jpg' };
}
