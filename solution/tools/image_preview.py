"""Photo preview that works with headless OpenCV builds."""
from pathlib import Path


def prediction_lines(result):
    """Show the emitted values, including the final Russian format profile."""
    predictions = result.get('predictions', [])
    if not predictions:
        return 'Номер найден, но не распознан' if result.get('detections') else 'Номер не обнаружен'
    return '\n'.join(
        f"{index}. {row['plate_num']}   {row['plate_type']}   confidence: {row['confidence']:.2f}"
        for index, row in enumerate(predictions, start=1)
    )


class ImagePreview:
    def __init__(self):
        try:
            import tkinter as tk
            from PIL import Image, ImageDraw, ImageTk
        except ImportError as error:
            raise ValueError('--show requires Tkinter and Pillow in the selected Python runtime') from error
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            raise ValueError('Cannot open the photo viewer: ' + str(error)) from error
        self.root.withdraw()
        self.root.title('Распознавание номеров')
        self.root.resizable(False, False)
        self.enabled = True
        self.advance = tk.BooleanVar(self.root, False)
        self.Image, self.ImageDraw, self.ImageTk = Image, ImageDraw, ImageTk
        self.max_size = (min(1200, max(200, self.root.winfo_screenwidth() - 100)),
                         min(720, max(150, self.root.winfo_screenheight() - 300)))
        self.heading = tk.Label(self.root, font=('Segoe UI', 12, 'bold'), anchor='w')
        self.heading.pack(fill='x', padx=12, pady=(10, 4))
        self.picture = tk.Label(self.root, background='#202020')
        self.picture.pack(padx=12)
        self.results = tk.Label(self.root, font=('Segoe UI', 12), justify='left', anchor='w')
        self.results.pack(fill='x', padx=12, pady=8)
        buttons = tk.Frame(self.root)
        buttons.pack(pady=(0, 10))
        tk.Button(buttons, text='Следующее фото', command=self._next).pack(side='left', padx=6)
        tk.Button(buttons, text='Закрыть просмотр', command=self._disable).pack(side='left', padx=6)
        tk.Label(self.root, text='Любая клавиша: следующее фото. Esc: продолжить без окон.').pack(pady=(0, 8))
        self.root.bind('<Key>', self._key)
        self.root.protocol('WM_DELETE_WINDOW', self._disable)

    def _next(self, event=None):
        self.advance.set(True)

    def _disable(self):
        self.enabled = False
        self.advance.set(True)
        self.root.withdraw()

    def _key(self, event):
        if event.keysym == 'Escape':
            self._disable()
        else:
            self._next()
        return 'break'

    def show(self, image, result, path, index, total):
        if not self.enabled:
            return
        # The recognition frame is BGR. Draw only on a separate RGB preview.
        rendered = self.Image.fromarray(image[:, :, ::-1])
        rendered.thumbnail(self.max_size, self.Image.Resampling.LANCZOS)
        sx, sy = rendered.width / image.shape[1], rendered.height / image.shape[0]
        draw = self.ImageDraw.Draw(rendered)
        emitted = {row['detection_index']: row for row in result.get('predictions', [])}
        for detection in result.get('detections', []):
            bbox = detection.get('bbox')
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (round(v * scale) for v, scale in zip(bbox, (sx, sy, sx, sy)))
            row = emitted.get(detection.get('detection_index'))
            color = '#39e75f' if row else '#ffae42'
            label = row['plate_num'] if row else '?'
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            box = draw.textbbox((0, 0), label)
            width, height = box[2] - box[0] + 8, box[3] - box[1] + 8
            left = min(max(0, x1), max(0, rendered.width - width))
            top = max(0, y1 - height)
            draw.rectangle((left, top, left + width, top + height), fill='#202020')
            draw.text((left + 4, top + 4 - box[1]), label, fill=color)
        self.photo = self.ImageTk.PhotoImage(rendered, master=self.root)
        self.picture.configure(image=self.photo)
        self.heading.configure(text=f'{index}/{total}   {Path(path).name}')
        self.results.configure(text=prediction_lines(result))
        self.advance.set(False)
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.wait_variable(self.advance)
        self.root.withdraw()

    def close(self):
        self.root.destroy()
