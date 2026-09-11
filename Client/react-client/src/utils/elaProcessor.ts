/**
 * @license
 * Заглушка (Mock) для ELA-анализа. 
 * Имитирует работу бэкенда для демонстрации интерфейса.
 */

// 1. Генерация фейкового SHA-256 хеша для красивого отображения в панели
export function generateFileHash(filename: string, fileSize: number): string {
  const pseudoRandom = Math.abs(filename.length * fileSize * Math.random()).toString(16).padStart(8, '0');
  return `sha256_${pseudoRandom}c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.slice(0, 64);
}

// 2. Имитация обработки картинки нейросетью
export function performELA(
  imageElement: HTMLImageElement,
  quality: number = 0.95,
  multiplier: number = 15
): Promise<{ elaDataUrl: string; heatmapDataUrl: string }> {
  return new Promise((resolve) => {
    // Искусственная задержка (1.5 секунды), чтобы показать красивый лоадер "ANALYZING MATRIX..."
    setTimeout(() => {
      const width = imageElement.naturalWidth || imageElement.width || 800;
      const height = imageElement.naturalHeight || imageElement.height || 600;

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        resolve({ elaDataUrl: '', heatmapDataUrl: '' });
        return;
      }

      // --- Имитация ELA-карты (Рисуем оригинал и сильно затемняем его) ---
      ctx.drawImage(imageElement, 0, 0, width, height);
      ctx.fillStyle = 'rgba(5, 10, 21, 0.85)'; // Цвет Void Navy с 85% непрозрачностью
      ctx.fillRect(0, 0, width, height);
      
      // Добавим немного красного "шума" по центру для наглядности шторки
      ctx.fillStyle = 'rgba(239, 68, 68, 0.3)'; // Terminal Red
      ctx.beginPath();
      ctx.arc(width / 2, height / 2, Math.min(width, height) * 0.2, 0, Math.PI * 2);
      ctx.fill();
      
      const elaMockUrl = canvas.toDataURL('image/png');

      // --- Имитация AI Heatmap (Рисуем оригинал и накладываем сине-красный тепловой фильтр) ---
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(imageElement, 0, 0, width, height);
      ctx.fillStyle = 'rgba(56, 189, 248, 0.4)'; // Electric Blue фильтр
      ctx.fillRect(0, 0, width, height);
      
      ctx.fillStyle = 'rgba(239, 68, 68, 0.6)'; // Terminal Red пятно
      ctx.beginPath();
      ctx.arc(width / 2, height / 2, Math.min(width, height) * 0.15, 0, Math.PI * 2);
      ctx.fill();

      const heatmapMockUrl = canvas.toDataURL('image/png');

      // Возвращаем готовые URL картинок
      resolve({
        elaDataUrl: elaMockUrl,
        heatmapDataUrl: heatmapMockUrl,
      });
    }, 1500); // 1.5 секунды задержки
  });
}