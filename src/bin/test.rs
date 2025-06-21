// OpenCV Rust Lane Detection with Debugging
// export OPENCV_VIDEOIO_PRIORITY_LIST=V4L2,GSTREAMER

use opencv::{
    core::{self, AlgorithmHint, Point, Scalar, Size, Vector},
    highgui, imgproc,
    prelude::*,
    videoio, Result,
};

fn main() -> Result<()> {
    lane_detection_camera()
}

// 차선 검출 메인 함수
fn lane_detection_camera() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    if !cam.is_opened()? {
        panic!("Cannot open camera!");
    }

    // 윈도우 생성
    highgui::named_window("Original", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Gray + Blur", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Edges", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("ROI", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Lane Detection", highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut frame_count = 0;

    println!("Lane detection started!");
    println!("Controls:");
    println!("  ESC - Exit");
    println!("  SPACE - Toggle debug info");
    println!("================================");

    loop {
        cam.read(&mut frame)?;

        if frame.empty() {
            continue;
        }

        frame_count += 1;

        // 차선 검출 처리
        let result = process_lane_detection_with_debug(&frame, frame_count)?;

        highgui::imshow("Original", &frame)?;
        highgui::imshow("Lane Detection", &result.final_image)?;
        highgui::imshow("Gray + Blur", &result.processed)?;
        highgui::imshow("Edges", &result.edges)?;
        highgui::imshow("ROI", &result.roi)?;

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

// 디버깅 정보를 포함한 결과 구조체
struct LaneDetectionResult {
    final_image: Mat,
    processed: Mat,
    edges: Mat,
    roi: Mat,
    lines_detected: usize,
    left_lanes: usize,
    right_lanes: usize,
}

// 차선 검출 처리 함수 (디버깅 포함)
fn process_lane_detection_with_debug(frame: &Mat, frame_count: i32) -> Result<LaneDetectionResult> {
    // 1. 그레이스케일 변환
    let mut gray = Mat::default();
    imgproc::cvt_color_def(frame, &mut gray, imgproc::COLOR_BGR2GRAY)?;

    // 2. 가우시안 블러로 노이즈 제거
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &gray,
        &mut blurred,
        Size::new(5, 5),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // 3. Canny 엣지 검출 (임계값 조정)
    let mut edges = Mat::default();
    imgproc::canny(&blurred, &mut edges, 50.0, 150.0, 3, false)?;

    // 4. 관심영역(ROI) 적용
    let roi_edges = apply_roi(&edges)?;

    // 5. 허프 변환으로 직선 검출
    let lines = detect_lines(&roi_edges)?;

    // 6. 차선 분류 및 그리기
    let (final_image, left_count, right_count) = draw_lanes_with_debug(frame, &lines)?;

    // 7. 디버깅 정보 출력 (10프레임마다)
    if frame_count % 30 == 0 {
        println!(
            "Frame {}: Lines detected: {}, Left: {}, Right: {}",
            frame_count,
            lines.len(),
            left_count,
            right_count
        );
    }

    Ok(LaneDetectionResult {
        final_image,
        processed: blurred,
        edges,
        roi: roi_edges,
        lines_detected: lines.len(),
        left_lanes: left_count,
        right_lanes: right_count,
    })
}

// 관심영역(ROI) 적용 - 간단한 사다리꼴
fn apply_roi(edges: &Mat) -> Result<Mat> {
    let height = edges.rows();
    let width = edges.cols();

    // ROI 영역 정의 (더 넓은 사다리꼴)
    let vertices = vec![
        Point::new(width / 10, height),            // 좌하단
        Point::new(width * 2 / 5, height * 3 / 5), // 좌상단
        Point::new(width * 3 / 5, height * 3 / 5), // 우상단
        Point::new(width * 9 / 10, height),        // 우하단
    ];

    // 마스크 생성
    let mut mask = Mat::zeros(edges.rows(), edges.cols(), core::CV_8UC1)?.to_mat()?;
    let points = Vector::<Point>::from_iter(vertices);
    let pts = Vector::<Vector<Point>>::from_iter(vec![points]);

    imgproc::fill_poly(
        &mut mask,
        &pts,
        Scalar::new(255.0, 0.0, 0.0, 0.0),
        imgproc::LINE_8,
        0,
        Point::new(0, 0),
    )?;

    // ROI 적용
    let mut result = Mat::default();
    core::bitwise_and(edges, &mask, &mut result, &core::no_array())?;

    Ok(result)
}

// 허프 변환으로 직선 검출 (파라미터 조정)
fn detect_lines(edges: &Mat) -> Result<Vector<core::Vec4i>> {
    let mut lines = Vector::<core::Vec4i>::new();

    imgproc::hough_lines_p(
        edges,
        &mut lines,
        2.0,                          // rho (거리 해상도)
        std::f64::consts::PI / 180.0, // theta (각도 해상도)
        30,                           // threshold (낮춤)
        50.0,                         // minLineLength (낮춤)
        20.0,                         // maxLineGap (낮춤)
    )?;

    Ok(lines)
}

// 차선 그리기 (디버깅 정보 포함)
fn draw_lanes_with_debug(image: &Mat, lines: &Vector<core::Vec4i>) -> Result<(Mat, usize, usize)> {
    let mut result = image.clone();
    let mut left_lines = Vec::new();
    let mut right_lines = Vec::new();

    let width = image.cols();
    let center_x = width / 2;

    // 모든 검출된 직선을 연한 색으로 그리기 (디버깅용)
    for i in 0..lines.len() {
        let line = lines.get(i)?;
        let x1 = line[0];
        let y1 = line[1];
        let x2 = line[2];
        let y2 = line[3];

        // 모든 직선을 회색으로 표시
        imgproc::line(
            &mut result,
            Point::new(x1, y1),
            Point::new(x2, y2),
            Scalar::new(128.0, 128.0, 128.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        if x2 - x1 == 0 {
            continue; // 수직선 방지
        }

        let slope = (y2 - y1) as f64 / (x2 - x1) as f64;
        let line_center_x = (x1 + x2) / 2;

        // 기울기와 위치로 좌우 분류 (더 엄격한 조건)
        if slope < -0.3 && line_center_x < center_x {
            // 좌측 차선 (음의 기울기, 화면 좌측)
            left_lines.push((x1, y1, x2, y2, slope));
        } else if slope > 0.3 && line_center_x > center_x {
            // 우측 차선 (양의 기울기, 화면 우측)
            right_lines.push((x1, y1, x2, y2, slope));
        }
    }

    // 좌측 차선 평균화 및 그리기
    if !left_lines.is_empty() {
        let avg_line = average_lines(&left_lines, image.rows());
        if let Some((x1, y1, x2, y2)) = avg_line {
            imgproc::line(
                &mut result,
                Point::new(x1, y1),
                Point::new(x2, y2),
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                8,
                imgproc::LINE_8,
                0,
            )?;
        }
    }

    // 우측 차선 평균화 및 그리기
    if !right_lines.is_empty() {
        let avg_line = average_lines(&right_lines, image.rows());
        if let Some((x1, y1, x2, y2)) = avg_line {
            imgproc::line(
                &mut result,
                Point::new(x1, y1),
                Point::new(x2, y2),
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                8,
                imgproc::LINE_8,
                0,
            )?;
        }
    }

    // 디버깅 텍스트 추가
    let debug_text = format!(
        "Lines: {} | Left: {} | Right: {}",
        lines.len(),
        left_lines.len(),
        right_lines.len()
    );

    imgproc::put_text(
        &mut result,
        &debug_text,
        Point::new(10, 30),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.7,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;

    Ok((result, left_lines.len(), right_lines.len()))
}

// 여러 직선의 평균 계산 (개선된 버전)
fn average_lines(lines: &[(i32, i32, i32, i32, f64)], height: i32) -> Option<(i32, i32, i32, i32)> {
    if lines.is_empty() {
        return None;
    }

    // 기울기 필터링 (극단적인 기울기 제거)
    let filtered_lines: Vec<_> = lines
        .iter()
        .filter(|(_, _, _, _, slope)| slope.abs() > 0.3 && slope.abs() < 3.0)
        .collect();

    if filtered_lines.is_empty() {
        return None;
    }

    let mut x_coords = Vec::new();
    let mut y_coords = Vec::new();

    for &(x1, y1, x2, y2, _) in &filtered_lines {
        x_coords.extend_from_slice(&[x1, x2]);
        y_coords.extend_from_slice(&[y1, y2]);
    }

    // 최소제곱법으로 직선 피팅
    let n = x_coords.len() as f64;
    if n < 2.0 {
        return None;
    }

    let sum_x: f64 = x_coords.iter().map(|&x| *x as f64).sum();
    let sum_y: f64 = y_coords.iter().map(|&y| *y as f64).sum();
    let sum_xy: f64 = x_coords
        .iter()
        .zip(y_coords.iter())
        .map(|(&x, &y)| *x as f64 * *y as f64)
        .sum();
    let sum_x2: f64 = x_coords.iter().map(|&x| (*x as f64).powi(2)).sum();

    let denominator = n * sum_x2 - sum_x.powi(2);
    if denominator.abs() < 1e-6 {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    let intercept = (sum_y - slope * sum_x) / n;

    // y = mx + b를 x = (y - b) / m로 변환
    if slope.abs() < 1e-6 {
        return None;
    }

    let y1 = height;
    let y2 = (height as f64 * 0.6) as i32;
    let x1 = ((y1 as f64 - intercept) / slope) as i32;
    let x2 = ((y2 as f64 - intercept) / slope) as i32;

    // 유효한 범위 체크
    if x1 >= 0 && x2 >= 0 && x1 < 2000 && x2 < 2000 {
        Some((x1, y1, x2, y2))
    } else {
        None
    }
}
