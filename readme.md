# Pictures 2 Panning Videos: Convert a Landscape Image into a Panning Video

This project allows you to convert a landscape image into a panning video. The output video is in 16:9 format, which is useful for Instagram stories, TikTok, and other social media platforms that support videos in this format.

## Features

- Convert a landscape image into a panning video
- Easy to use
- High-quality output
- Customizable output size, frame rate, and movement direction
- Add audio to the output video
- Support for multiple images
- Support for multiple audio formats
- Support for multiple movement directions
- Very fast

## Installation

```bash
git clone https://github.com/affromero/pic2panning.git
poetry install
```

## Usage CLI

```bash
poetry run python -m pic2panning.main --images https://images.pexels.com/photos/3125171/pexels-photo-3125171.jpeg --output_file output.mp4 --time 5 --ratio 16:9 --audio.files https://www.youtube.com/watch?v=dQw4w9WgXcQ --output_size 1080 1920 --fps 240 --movement panning-lr
```

## Demos

### Example Panning from left to right

<img src="https://images.pexels.com/photos/29188556/pexels-photo-29188556/free-photo-of-stunning-sunset-over-mulafossur-waterfall-faroe-islands.jpeg" alt="Example Panning" width="500"/>

<video src='assets/demo_panning.mp4' width=250/> | <video src='assets/demo_panning_focus.mp4' width=250/>

### Example Zooming in and out

<img src="https://images.pexels.com/photos/2113566/pexels-photo-2113566.jpeg" alt="Example Zoom" width="500"/>

<video src='assets/demo_zoom.mp4' width=500/>

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License
