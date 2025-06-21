// OpenCV Rust
// export OPENCV_VIDEOIO_PRIORITY_LIST=V4L2,GSTREAMER

use opencv::{
    core::{self, AlgorithmHint},
    highgui, imgproc,
    prelude::*,
    videoio, Result,
};

fn main() -> Result<()> {
    // Basic camera stream
    basic_camera()
    //grayscale_example()
    //edge_detection_example()
    //blur_example()
    //color_filter_example()
}

// Basic camera (working version)
fn basic_camera() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    //let mut cam = videoio::VideoCapture::new(0, videoio::CAP_V4L2) // Ubuntu setup
    //    .or_else(|_| videoio::VideoCapture::new(0, videoio::CAP_ANY))?;
    if !cam.is_opened()? {
        panic!("Cannot open camera!");
    }

    highgui::named_window("Camera", highgui::WINDOW_AUTOSIZE)?;
    let mut frame = Mat::default();

    println!("Camera started! Press ESC to exit");

    loop {
        cam.read(&mut frame)?;

        if frame.empty() {
            continue;
        }

        highgui::imshow("Camera", &frame)?;

        let key = highgui::wait_key(30)?;
        if key == 27 {
            // ESC key
            break;
        }
    }

    cam.release()?;
    highgui::destroy_all_windows()?;
    Ok(())
}

// Grayscale conversion example (correct AlgorithmHint usage)
#[allow(dead_code)]
fn grayscale_example() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    if !cam.is_opened()? {
        panic!("Cannot open camera!");
    }

    highgui::named_window("Grayscale", highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut gray = Mat::default();

    println!("Grayscale camera started! Press ESC to exit");

    loop {
        cam.read(&mut frame)?;

        if !frame.empty() {
            // Method 1: Simple approach (recommended)
            imgproc::cvt_color_def(&frame, &mut gray, imgproc::COLOR_BGR2GRAY)?;

            // Method 2: Full parameters (correct AlgorithmHint value)
            //imgproc::cvt_color(
            //    &frame,
            //    &mut gray,
            //    imgproc::COLOR_BGR2GRAY,
            //    0,
            //    AlgorithmHint::ALGO_HINT_DEFAULT, // correct value
            //)?;

            highgui::imshow("Grayscale", &gray)?;
        }

        if highgui::wait_key(30)? == 27 {
            break;
        }
    }

    cam.release()?;
    highgui::destroy_all_windows()?;
    Ok(())
}

// Canny edge detection example (parameter count fixed)
#[allow(dead_code)]
fn edge_detection_example() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    if !cam.is_opened()? {
        panic!("Cannot open camera!");
    }

    highgui::named_window("Edge Detection", highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut gray = Mat::default();
    let mut edges = Mat::default();

    println!("Edge detection started! Press ESC to exit");

    loop {
        cam.read(&mut frame)?;

        if !frame.empty() {
            // Convert to grayscale
            imgproc::cvt_color_def(&frame, &mut gray, imgproc::COLOR_BGR2GRAY)?;

            // Canny edge detection (using only 6 parameters)
            imgproc::canny(&gray, &mut edges, 50.0, 150.0, 3, false)?;

            highgui::imshow("Edge Detection", &edges)?;
        }

        if highgui::wait_key(30)? == 27 {
            break;
        }
    }

    cam.release()?;
    highgui::destroy_all_windows()?;
    Ok(())
}

// Blur effect example
#[allow(dead_code)]
fn blur_example() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    if !cam.is_opened()? {
        panic!("Cannot open camera!");
    }

    highgui::named_window("Blur Effect", highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut blurred = Mat::default();

    println!("Blur effect started! Press ESC to exit");

    loop {
        cam.read(&mut frame)?;

        if !frame.empty() {
            // Simple blur effect
            imgproc::blur(
                &frame,
                &mut blurred,
                core::Size::new(15, 15),
                core::Point::new(-1, -1),
                core::BORDER_DEFAULT,
            )?;

            highgui::imshow("Blur Effect", &blurred)?;
        }

        if highgui::wait_key(30)? == 27 {
            break;
        }
    }

    cam.release()?;
    highgui::destroy_all_windows()?;
    Ok(())
}

// Color filter example
#[allow(dead_code)]
fn color_filter_example() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    if !cam.is_opened()? {
        panic!("Cannot open camera!");
    }

    highgui::named_window("Color Filter", highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut hsv = Mat::default();
    let mut mask = Mat::default();
    let mut result = Mat::default();

    // Blue color range (HSV)
    let lower_blue = core::Scalar::new(100.0, 50.0, 50.0, 0.0);
    let upper_blue = core::Scalar::new(130.0, 255.0, 255.0, 0.0);

    println!("Blue color filter started! Press ESC to exit");

    loop {
        cam.read(&mut frame)?;

        if !frame.empty() {
            // Convert BGR to HSV
            imgproc::cvt_color_def(&frame, &mut hsv, imgproc::COLOR_BGR2HSV)?;

            // Create mask for blue regions
            core::in_range(&hsv, &lower_blue, &upper_blue, &mut mask)?;

            // Apply mask to original image
            core::bitwise_and(&frame, &frame, &mut result, &mask)?;

            highgui::imshow("Color Filter", &result)?;
        }

        if highgui::wait_key(30)? == 27 {
            break;
        }
    }

    cam.release()?;
    highgui::destroy_all_windows()?;
    Ok(())
}
