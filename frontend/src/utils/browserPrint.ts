import apiClient from '../apiClient';

/**
 * Print documents with the browser's own print dialog. Used when the backend has no physical
 * printer (cloud deployment): the backend returns a document URL and the user picks the printer.
 */

export interface BrowserPrintItem {
  /** Path on the API, e.g. /document/processed/scan.png */
  url: string;
  copies?: number;
}

const waitForImages = async (doc: Document): Promise<void> => {
  await Promise.all(
    Array.from(doc.images).map((img) =>
      img.complete ? Promise.resolve() : new Promise<void>((resolve) => { img.onload = img.onerror = () => resolve(); })
    )
  );
};

const fetchBlob = async (url: string): Promise<Blob> => {
  const response = await apiClient.get<Blob>(url, { responseType: 'blob', timeout: 60000 });
  return response.data;
};

const printInIframe = (build: (iframe: HTMLIFrameElement) => void, ready: (iframe: HTMLIFrameElement) => Promise<void>) =>
  new Promise<void>((resolve) => {
    const iframe = document.createElement('iframe');
    iframe.style.cssText = 'position:fixed;right:0;bottom:0;width:0;height:0;border:0;visibility:hidden';
    document.body.appendChild(iframe);
    const cleanup = () => setTimeout(() => iframe.remove(), 60000);
    iframe.onload = async () => {
      try {
        await ready(iframe);
        iframe.contentWindow?.focus();
        iframe.contentWindow?.print();
      } finally {
        cleanup();
        resolve();
      }
    };
    build(iframe);
  });

/** Opens the browser print dialog for the given documents. Resolves once the dialog was requested. */
export async function printWithBrowser(items: BrowserPrintItem[]): Promise<void> {
  const blobs = await Promise.all(items.map(async (item) => ({ blob: await fetchBlob(item.url), copies: Math.max(1, item.copies || 1) })));
  const images = blobs.filter((entry) => entry.blob.type.startsWith('image/'));
  const others = blobs.filter((entry) => !entry.blob.type.startsWith('image/'));

  if (images.length > 0) {
    const urls = images.flatMap((entry) => Array.from({ length: entry.copies }, () => URL.createObjectURL(entry.blob)));
    const html =
      '<!doctype html><html><head><style>@page{margin:0}html,body{margin:0}' +
      'img{display:block;width:100%;height:auto;max-height:100vh;object-fit:contain;page-break-after:always}</style></head><body>' +
      urls.map((src) => `<img src="${src}">`).join('') +
      '</body></html>';
    await printInIframe(
      (iframe) => { iframe.srcdoc = html; },
      (iframe) => waitForImages(iframe.contentDocument as Document)
    );
    setTimeout(() => urls.forEach((src) => URL.revokeObjectURL(src)), 120000);
  }

  // PDFs print from the browser's built-in viewer. If the browser blocks printing from a hidden frame,
  // open the file so the user can print it from the viewer.
  for (const entry of others) {
    const src = URL.createObjectURL(entry.blob);
    try {
      await printInIframe(
        (iframe) => { iframe.src = src; },
        () => new Promise((resolve) => setTimeout(resolve, 800))
      );
    } catch {
      window.open(src, '_blank');
    }
    setTimeout(() => URL.revokeObjectURL(src), 120000);
  }
}
