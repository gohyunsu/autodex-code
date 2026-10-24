# 여러 카메라에서 동시에 보는 키 끝면 중심과 소켓 축

저장 위치: `/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/methods/sam_half_ellipse/`.

`animations_multiview/`의 GIF는 한 에피소드의 저장된 `preinsert` 프레임을 **카메라별로 가로로 나열**한다. 각 단계에서 모든 카메라를 동시에 보여준다. 2뷰·3뷰·5뷰 에피소드를 각각 저장했다. **각 카메라의 원본 512 × 512 크롭을 여섯 단계 내내 동일하게 유지**하고 그 위에 표시 요소만 순서대로 더한다. 각 단계에는 단계명과 카메라 ID를, **5·6단계 하단의 공통 영역 한 곳**에는 소켓 좌표계의 키 추정점과 축까지의 XY 변위를 mm로 표시한다. `animation_frames_multiview/`에는 GIF의 여섯 단계가 PNG로 따로 있다.

| 단계 | 표시 내용 |
| --- | --- |
| 1 | 저장된 undistorted 카메라 crop 원본 |
| 2 | Qwen이 찾은 흰색 키의 박스 |
| 3 | Qwen 박스 범위에서 얻은 SAM3 키 마스크 |
| 4 | SAM3의 아래쪽 윤곽에 맞춘 **관측 반타원과 중심** |
| 5 | 모든 카메라의 관측 중심을 삼각화한 **하나의 3D proxy**를 각 카메라에 재투영한 점. 측정 소켓 포즈의 입구 고리와 축도 표시 |
| 6 | 같은 3D proxy 높이에서 **소켓 축에 있는 점**과, proxy에서 그 점으로 향하는 XY 변위의 카메라별 투영. 소켓 축도 유지 |

5·6단계에서 초록색은 SAM3 관측 타원, 분홍색 점은 다중 뷰 3D proxy의 재투영, 청록색 고리는 측정 소켓 입구다. **청록색 실선은 입구 고리의 3D 중심에서 시작하는 측정 소켓 자세의 국소 +Z 축**을 각 카메라에 투영한 것이다. 고리와 축은 같은 색과 3 px 선 두께로 그렸다. 표시 구간은 소켓 입구 높이부터 입구 위 최소 90 mm까지이며, proxy 높이보다 20 mm 이상 높게 끝나도록 연장한다. 이 구간은 축을 눈으로 읽기 위한 그림 범위이지 소켓 실물 길이나 삽입 명령 거리가 아니다. 6단계의 노란 점은 아래 식의 소켓 축 목표 위치이며 주황색 화살표는 분홍색 점에서 노란 점으로 향한다. 그림의 점과 화살표는 촬영 원본에 더한 주석이다.

소켓 좌표계에서 추정한 키 끝면 중심을 `p=(x,y,z)`라고 할 때, 그림의 노란 점은 `p_axis=(0,0,z)`이다. 따라서 표시한 변위는 `Δ_socket=(-x,-y,0)`이며 **높이와 자세는 바꾸지 않는 XY 기하 비교**다. 실제 로봇 관절 경로, 충돌 회피, 힘 제어, 다음 시행의 보정량을 이 화살표만으로 결정하지 않는다. 실루엣에서 얻은 3D 점은 실제 물리적 면 중심의 계측 정답이 아니다.

GIF 하단의 `key proxy`는 이 추정점의 `(x,y,z)`를 mm로 쓴 값이다. `XY to axis`는 `(-x,-y)`이고 `distance`는 `sqrt(x²+y²)`이다. 모두 **소켓 좌표계**의 수치다. 화면의 주황 화살표는 이 3D 수평 변위를 해당 카메라 영상에 투영한 것이므로 화면상 기울기는 뷰마다 다르다.

| 에피소드 | 동시 표시 뷰 | 그림의 Δ_socket XY (mm) |
| --- | ---: | ---: |
| `20261022_101456_545928` | 2 | (−16.48, −3.04) |
| `20261022_103429_752726` | 5 | (−16.80, −4.56) |
| `20261022_105828_126084` | 3 | (−31.23, −9.53) |

이 수치는 저장 사진에서 얻은 **시각적 proxy의 기하 변위**다. 크기가 실제 삽입 오차로 검증된 값은 아니며 로봇에 보내지 않았다. 소켓 포즈 투영이 눈으로 맞아 보이는 세 에피소드만 사용했다. 제외 사유는 상위 폴더의 `socket_pose_visual_qc.json`, 카메라별 frame ID·timestamp·원본 SHA-256·마스크 SHA-256, 축 구간과 점 좌표는 `multiview_xy_animation.json`에 있다. 이전 SAM3 마스크는 상위 폴더의 `inputs/`로 정리했으며, 각 파일을 다시 확인했는지는 `mask_file_reverified_at_animation`에 기록했다.

- [2뷰 GIF](/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/methods/sam_half_ellipse/animations_multiview/20261022_101456_545928.gif) · [마지막 PNG](/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/methods/sam_half_ellipse/animation_frames_multiview/20261022_101456_545928/06_xy_alignment.png)
- [5뷰 GIF](/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/methods/sam_half_ellipse/animations_multiview/20261022_103429_752726.gif) · [마지막 PNG](/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/methods/sam_half_ellipse/animation_frames_multiview/20261022_103429_752726/06_xy_alignment.png)
- [3뷰 GIF](/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/methods/sam_half_ellipse/animations_multiview/20261022_105828_126084.gif) · [마지막 PNG](/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/methods/sam_half_ellipse/animation_frames_multiview/20261022_105828_126084/06_xy_alignment.png)

재생 코드: `render_saved_key_face_multiview_gif.py`. 입력은 상위 폴더의 기존 비교 결과와 실제 촬영 원본이다. 스크립트는 카메라 원본 해시와 프레임 ID를 확인하고, SAM3 마스크 파일이 남아 있으면 마스크 해시도 다시 확인한다. 소켓 축은 각 에피소드의 고정 세션 보정에 저장된 `socket_pose_robot`와 해당 소켓 규격의 `task_geometry.json`을 사용한다. 이 형상 파일의 해시 확인된 사본은 상위 폴더의 `inputs/fixture_assets/`에 있으므로 재생 시 `--fixture-root`로 그 위치를 지정한다. 카메라나 로봇에 연결하지 않는다.
