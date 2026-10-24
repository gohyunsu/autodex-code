# 저장된 pre-insertion 사진에서 키 끝면 중심의 시각적 proxy

이 실험은 실제 AutoDex 에피소드에 저장된 **undistorted 원본 PNG**만 다시 읽는다. 카메라 연결, 로봇 명령, 다음 시행의 XY 갱신은 수행하지 않는다. 이전 검토의 출력 그림과 JSON은 `/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/archive/previous_key_tip_review/`에 정리했다. 모든 점은 추정치이며, 실제 키 끝면의 3D 정답이나 로봇 목표점으로 사용하지 않았다.

## 입력과 단계

| 단계 | 입력 | 처리 | 출력 |
| --- | --- | --- | --- |
| 1. 원본 연결 | 에피소드 raw preinsert PNG, frame ID, camera ID, timestamp, SHA-256 | 저장된 PNG의 해시와 프레임 ID를 Qwen·SAM3·3D 결과 사이에서 대조 | 근거가 같은 카메라 프레임 |
| 2. 크롭 | frozen session `K_undist`, 카메라 extrinsic, C2R, postlift 계획 키 포즈, 키 길이 | 계획 키 축의 양 끝을 투영하여 원본 픽셀의 512×512 창 선택. 계획 축은 검색 위치와 원위단 방향만 제공 | 원본 해상도의 key-axis crop |
| 3. Qwen | 크롭, 로컬 Qwen3-VL-2B | 아래 영어 프롬프트로 키 전체 bounding box 반환. 0–1000 좌표를 crop 픽셀로 변환 | 키 대략적 범위 |
| 4. SAM3 | Qwen 박스에서 25px 여유를 둔 원본 crop, AutoDex의 SAM3 image processor | `white cylinder` 프롬프트로 키 마스크 추정 | 이진 키 실루엣 |
| 5. 끝 경계 | SAM3 마스크, 계획 축의 투영 방향 | 기존 후보: 마스크 전체 회전 사각형의 원위단 변 중앙. 새 후보: 이 회전 사각형의 **가로 중심선은 유지**하고, 중앙 폭의 8개 구간에서 실제 끝 윤곽(축 방향 99백분위)을 샘플링해 robust 선형 적합; 그 중심선에서 축 방향 끝 위치만 보정 | 두 2D 끝점 후보, 윤곽 샘플, 적합선 |
| 6. 3D proxy | 뷰별 2D 끝점, frozen socket pose, undistorted 다중 카메라 투영 | 카메라 두 개 이상으로 socket 좌표계에서 robust least-squares 삼각화, 각 뷰로 재투영 | socket-frame 3D proxy, 카메라별 픽셀 잔차 |

Qwen 프롬프트는 다음과 같다.

```text
Find the WHITE CYLINDRICAL KEY directly held by the BLACK ROBOT HAND. Ignore any white socket, board, or fingers. Return the tight bounding box of the white cylindrical KEY ONLY as JSON {"bbox":[x1,y1,x2,y2]}. Coordinates are normalized 0-1000 for the whole displayed image.
```

SAM3 텍스트 프롬프트는 `white cylinder`이다. 소켓 역시 흰색이므로 Qwen 박스와 계획 키 축 크롭으로 SAM3 검색 영역을 좁힌다. 이 제한은 키의 관측 좌표를 계획값으로 고정하지 않는다.

## 저장된 실제 에피소드에서의 결과

아래 XY는 측정된 socket 축에서 **추정 키 끝점**까지의 socket-frame 거리다. 삽입 보정 명령이나 정확도 점수가 아니다. 픽셀 잔차는 같은 점의 다중 카메라 재투영 일관성만 나타낸다.

| 에피소드 | 사용 뷰 | 추정 XY (mm) | 재투영 잔차 중앙값 (px) | 해석 |
| --- | ---: | ---: | ---: | --- |
| `20261021_090945_053190` | 2 | (+9.2, +5.3) | <0.1 | 두 뷰의 마스크 경계 사용 |
| `20261021_104732_950124` | 2 | (+4.4, −22.6) | 5.0 | 한 뷰에서 SAM3 마스크가 없어 Qwen 박스 끝점으로 대체; 불확실 |
| `20261021_111900_186891` | 3 | (−20.8, −11.4) | 4.8 | 기울어진 키에서도 가로 중심선 유지 |
| `20261022_101456_545928` | 2 | (+15.9, +3.6) | 1.1 | 두 뷰의 마스크 경계 사용 |
| `20261022_103429_752726` | 5 | (+17.6, +4.7) | 3.9 | 추가 저장 뷰 다섯 개로 재생. 처음 선택된 한 뷰만으로는 3D 불가 |
| `20261022_105828_126084` | 3 | (+29.7, +9.8) | 6.3 | 세 뷰의 마스크 경계 사용 |

13뷰 기본 재생 중 SAM3가 12뷰에서 마스크를 반환했고, 103429 에피소드의 추가 5뷰에서는 모두 반환했다. 새 경계점은 기존 회전 사각형 끝점에서 중앙값 3.4px, 최대 5.0px 움직였다. 처음 시도한 몸통 가로 중앙 기반 적합은 기울어진 한 키에서 21.4px 옆으로 벗어나 버렸기 때문에 버리고, 기존 회전 사각형의 중심선을 유지하는 위 방법으로 수정했다. 비교 가능한 모든 에피소드에서 다중 뷰 재투영 잔차 중앙값은 감소했다. 이는 **서로 잘 맞는다는 증거**이고, 실제 끝면 중심에 더 가깝다는 증거는 아니다.

103429의 독립적인 사람 **대략 표시점** 4개와 비교하면, 기존→새 경계점 차이는 카메라별로 `24122734` 1.8→0.8px, `25305461` 19.8→19.6px, `25322638` 1.3→3.5px, `25322649` 2.2→2.7px이다. 따라서 이 자료만으로 새 점을 정답으로 선택할 수 없다. `25305463`의 끝면은 손·소켓과 겹쳐 사람 표시 정답이 없다. 원위단 면 자체가 가려지면 실루엣에서 물리적 면 중심을 유일하게 복원할 수 없다.

## 그림 읽기

- `process/*_method.png`: 실제 원본 → 계획 축 crop → Qwen 박스 → SAM3 마스크 → 실제 끝 경계의 노란 샘플·적합선 → 소켓 포즈와 3D proxy의 재투영. 흰 원은 이전 회전 사각형 방식의 끝점이다.
- `multiview/*_multiview.png`: 6개 에피소드의 기본 저장 뷰를 각각 모았다.
- `manual_multiview/20261022_103429_752726_multiview.png`: 같은 에피소드의 추가 저장 뷰 5개.
- 청록 원·XYZ는 frozen calibration으로 투영한 **측정 소켓 포즈**, 초록 점은 **계획 키 끝**, 분홍 박스·점은 **Qwen의 키 박스·아래 중앙**, 노란 점은 **마스크 경계 후보**, 보라 X는 **3D proxy의 재투영**이다. 103429 추가 뷰의 주황 X는 사람의 대략 표시점이며 정답 계측은 아니다. 그림의 유색 선·점은 원본 이미지의 픽셀이 아닌 주석이다.
- `process/process_metadata.json`, `process/process_metadata_manual.json`, `multiview/review_metadata.json`, `manual_multiview/review_metadata.json`에는 파일 경로, frame ID, timestamp, 원본 SHA-256, 프롬프트와 응답을 기록했다.

## 현재 파이프라인과의 관계

현재 실제 실행의 XY 진단은 `retry_session.py`에서 `vlm` 또는 `tracked_tip` 모드를 선택한다. 이 문서의 Qwen 박스 + SAM3 경계는 **저장 자료에 대한 읽기 전용 후보**로 구현했다. 실제 삽입 제어에 연결하거나 새 안전·실패 기준으로 사용하지 않았다. 기존 `tracked_tip`은 초기 tabletop 키 SAM 마스크에서 카메라별 색 특징을 배우고, preinsert 프레임의 계획 축 주변에서 키 끝을 찾는다. 옵션의 11프레임 짧은 영상 추적도 preinsert 근처에서 시작한다. 따라서 최초 FoundPose 6D 추정 시점부터 동일 픽셀을 지속 추적하는 방식은 아니다.

재생 코드: `held_key_tip.py`, `refine_saved_key_tip_boundary.py`, `triangulate_saved_key_tip.py`, `review_saved_key_tip_bbox.py`, `render_saved_key_tip_method.py`. 마스크 보정과 3D 계산은 저장된 이미지와 마스크만 다시 읽으며 모델을 재호출하지 않는다.
