# 저장된 실제 사진에서 키 아래 면 중심 추정

이 문서는 `/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/`에 저장된 **읽기 전용 재생**의 절차와 결과다. 입력은 AutoDex가 이미 촬영한 undistorted `preinsert` PNG 및 해당 세션의 카메라·소켓 보정이다. 카메라 연결, 로봇 동작, 새 에피소드 XY 명령은 없다. 그림의 선과 점은 원본 위에 계산 결과를 그린 주석이다.

## 파일 위치

이 폴더의 `inputs/`는 이번 Qwen·SAM3 입력 및 마스크와 소켓 규격별 `fixture_assets/` 스냅샷, `methods/`는 방법별 최종 결과와 카메라별 PNG, `role_images/`는 표시 목적별 멀티뷰 그리드다. `archive/`에는 이전 키 끝점·타원 검토본을 모았다. 이동 전후 절대경로와 10개 뷰의 원본·마스크 해시 검증 기록은 `asset_layout.json`에, 규격별 형상 파일의 복사 출처와 SHA-256은 `inputs/fixture_assets/PROVENANCE.json`에 있다. 기존 대형 보관본 `../20261023_prev/`는 그대로 두었다. 실제 AutoDex 촬영 원본과 에피소드 보정 파일은 해당 AutoDex 실험 디렉터리에 남아 있다.

## 두 타원의 뜻

- **초록색, 관측 타원:** Qwen이 키의 대략적인 박스를 찾고 SAM3가 분할한 **실제 영상 마스크의 아래쪽 윤곽**에 맞춘 타원이다. 그 중심을 키 아래 면 중심의 2D proxy로 선택했다. 이 적합에는 소켓 포즈, 계획 키의 원형 투영, 계획된 중심 좌표가 들어가지 않는다.
- **파란색, 예상 타원:** 키의 알려진 반지름 15 mm와 저장된 **계획 6D 키 자세**를 보정된 카메라에 투영한 원이다. 형상과 계획 위치를 보여주는 독립 진단 그림이다. 실제 관측점이 아니므로 다중 뷰 비교 점수 또는 3D proxy 계산에 넣지 않았다.
- 마지막 단계 그림의 **청록색 고리**는 저장된 **측정 소켓 포즈**에서 투영한 소켓 입구다. 이전 그림의 소켓 고리 색과 3 px 두께를 유지했다. 같은 색의 축 선은 그 입구 중심에서 시작한다. 초록색은 관측 키 타원, 분홍색 점은 다중 뷰 3D proxy를 해당 카메라에 재투영한 위치다.

| 역할 | 색 |
| --- | --- |
| Qwen 키 검색 박스 | 보라 `#b98aff` |
| SAM3 키 마스크·관측 아래 면 타원 | 초록 `#42f58d` |
| 타원 적합에 쓴 영상 경계 샘플 | 흰색 `#ffffff` |
| 다중 뷰 키 끝면 추정점 | 분홍 `#f85dc7` |
| 측정 소켓 입구 고리·입구 중심에서 시작하는 축 | 청록 `#00daf3` |
| 같은 높이의 소켓 축 목표점 | 노랑 `#ffe15b` |
| 추정점에서 축까지의 XY 화살표 | 주황 `#ff9f35` |
| 계획 키 자세 투영 | 파랑 `#7fa8ff` |
| 다른 끝점 추정 방법의 진단 그림 | 연주황 `#ffca80` |

두 타원은 모두 키를 나타내지만 관측값과 계획값의 역할이 달라 색을 분리했다. 이 팔레트는 `saved_key_face_palette.py` 한 곳에서 관리한다.

## 실제 에피소드에서 실행한 순서

1. **원본 연결.** 저장된 `preinsert` PNG의 camera ID, frame ID, timestamp, SHA-256을 Qwen 박스와 SAM3 마스크 결과에 맞춘다. 원본 해시가 달라지면 그림을 만들지 않는다.
2. **대략 위치.** 저장된 계획 키 축의 투영으로 원본 픽셀의 512 × 512 crop을 선택한다. 로컬 Qwen3-VL-2B에 아래 영어 프롬프트를 보내 키 전체 박스를 받았다. 계획 축은 검색 영역과 아래쪽 끝의 *방향*에만 사용한다.
3. **관측 마스크.** Qwen 박스를 25 px 넓힌 원본 crop에 기존 AutoDex SAM3 이미지 처리기를 적용하고 `white cylinder`로 키 실루엣을 얻었다. 소켓도 흰색이므로 박스로 검색 범위를 제한한다.
4. **관측 타원.** SAM3 마스크의 가장 큰 연결 영역을 골라 분리된 작은 흰 영역을 제외하고, 그 영역의 최소 회전 사각형 긴 축을 키 영상 축으로 사용한다. 계획 축은 두 끝 중 어느 쪽이 아래쪽인지 고르는 데만 쓴다. 몸통 중간 구간의 좌우 7·93 백분위에서 보이는 반지름을 추정한다. 반지름의 좌우 ±90%를 14개 구간으로 나눠 각 구간의 가장 아래쪽 마스크 경계(축 방향 99 백분위)를 모은다. 영상의 원근 투영을 단순화한 반타원식 `a = a_center + r_minor sqrt(1-(l/r_major)^2)`을 robust least-squares로 적합한다. 몸통 회전 사각형의 가로 중심선을 유지하고 `a_center`를 구해 **초록색 2D 중심**을 얻는다. 실제 원형 면 전체가 보이는 것으로 가정하지 않는다.
5. **다중 뷰 proxy.** 뷰마다 얻은 초록색 중심을 frozen `K_undist`, camera extrinsic, `c2r`, 측정 소켓 포즈로 만든 투영행렬에 넣어 socket 좌표계에서 robust least-squares로 한 점을 삼각화한다. 결과를 각 영상에 되그려 분홍색 점으로 표시한다. 이 점은 근사 대응점이며 로봇 목표나 정확한 실제 키 끝면 좌표가 아니다.

Qwen에 사용한 전체 프롬프트:

```text
Find the WHITE CYLINDRICAL KEY directly held by the BLACK ROBOT HAND. Ignore any white socket, board, or fingers. Return the tight bounding box of the white cylindrical KEY ONLY as JSON {"bbox":[x1,y1,x2,y2]}. Coordinates are normalized 0-1000 for the whole displayed image.
```

## 저장 자료 비교

소켓 입구 투영과 실제 흰 소켓 입구를 저장된 영상에서 육안 대조하여 `20261022_101456_545928`, `20261022_103429_752726`, `20261022_105828_126084`만 이 표에 넣었다. `20261021_090945_053190`, `20261021_104732_950124`, `20261021_111900_186891`은 소켓 포즈 투영이 어긋나 보이거나 확인 가능한 소켓 입구가 부족해 제외했다. 이 선별은 `socket_pose_visual_qc.json`의 **육안 스크리닝**이며 소켓 포즈의 정밀 계측 인증이 아니다.

다음 값은 카메라 사이에서 같은 후보점이 얼마나 일관되는지 나타내는 **픽셀 재투영 오차 중앙값**이다. 괄호는 3뷰 이상에서 한 뷰를 빼고 만든 3D 점을 빠진 뷰에 투영한 오차 중앙값이다. 낮은 값이 실제 물리적 면 중심에 더 가깝다는 보장은 없다. 두 뷰만 있을 때 한 뷰 제외 검증은 정의되지 않는다.

| 에피소드 | 뷰 수 | SAM 반타원 중심 | SAM 끝 경계점 | SAM 회전 사각형 끝 | 자유 OpenCV 타원 중심 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20261022_101456_545928` | 2 | **1.14 px** (불가) | 1.15 px (불가) | 1.53 px (불가) | 14.99 px (불가) |
| `20261022_103429_752726` | 5 | **1.78 px (2.76 px)** | 3.87 px (6.60 px) | 4.62 px (7.86 px) | 6.68 px (10.80 px) |
| `20261022_105828_126084` | 3 | **2.85 px (8.58 px)** | 6.26 px (19.31 px) | 6.72 px (20.82 px) | 4.57 px (9.96 px) |

이 세 에피소드의 **저장 자료 후보 중** SAM 반타원 중심을 선택했다. 5뷰 재투영과 한 뷰 제외 결과가 가장 좋은 증거다. 자유 `cv2.fitEllipse`를 아래쪽 윤곽 일부에 직접 적용한 방법은 잘린 윤곽·손 가림 때문에 중심이 불안정했다. 키의 아래 면이 소켓이나 손에 가려지거나 SAM3 마스크가 잘못되면 반타원의 중심도 틀릴 수 있다. 정답으로 계측한 3D 끝면 중심은 없어 물리적 정확도나 삽입 성공률은 여기서 판정할 수 없다. 이를 새로운 실패 게이트로 사용하지 않는다.

## 그림과 GIF

`methods/`에는 방법별 하위 폴더를 두었다. 각 방법 폴더의 `per_view/`에는 **카메라별 PNG 한 장씩** 저장했다. 파일명에는 에피소드 ID, 카메라 ID, 프레임 ID가 들어 있다. 그 밖에 세 에피소드의 멀티뷰 그리드, `results.json`, 상세 `README.md`가 있다. `sam_half_ellipse/animations/`에는 대표 뷰의 단계별 GIF도 있다. 계획 자세 원 투영은 관측 후보와 구분해 `planned_pose_projection_diagnostic/`에 두었다.

그 밖에 **그림의 역할**별 `role_images/`도 나눴다. 세 에피소드 모두 같은 이름의 PNG가 있다.

- `role_images/original/`: 실제 512 px 원본 crop만 표시.
- `role_images/qwen_box/`: Qwen이 찾은 키 박스만 표시.
- `role_images/sam_mask/`: SAM3 분할 영역과 윤곽을 초록색으로 표시.
- `role_images/observed_ellipse/`: SAM3 마스크 + 관측 타원·중심. 흰 작은 점은 실제 아래쪽 윤곽 샘플이다.
- `role_images/planned_projection/`: 계획 6D 키 자세의 30 mm 지름 원 투영을 파란색으로 표시. **관측 타원이 아니다.**
- `role_images/spatial_proxy/`: 관측 키 타원·다중 뷰 proxy의 재투영·측정 소켓 입구만 표시.
- `animations/`: 실제 에피소드별 5단계 GIF. 각 단계는 대표 카메라 `24122734`의 동일한 저장 프레임을 쓴다. GIF에는 단계명만 있다. `animation_stages/`에 GIF의 개별 프레임도 있다. 다른 카메라는 각 역할별 PNG에서 볼 수 있다.
- `methods/sam_half_ellipse/animations_multiview/`: 각 에피소드의 모든 저장 뷰를 가로로 동시에 보여주는 6단계 GIF. 마지막 두 단계에서 측정 소켓 자세의 축을 청록 실선으로, 다중 뷰 키 끝면 3D proxy와 같은 높이의 축 목표점 및 XY 화살표를 표시한다. 소켓 좌표계의 계산값은 모든 뷰 아래 공통 영역 한 곳에 mm로 표시한다. [계산과 색상 설명](/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/methods/sam_half_ellipse/MULTIVIEW_XY_ANIMATION.md)에 기하식과 한계를 적었다.

대표 사례: [카메라 24122734 한 장의 관측 타원](/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/methods/sam_half_ellipse/per_view/observed_ellipse/20261022_103429_752726_24122734_frame1761792.png), [5뷰 그리드](/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/methods/sam_half_ellipse/observed_ellipse/20261022_103429_752726.png), [별도 계획 타원](/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/methods/planned_pose_projection_diagnostic/planned_projection/20261022_103429_752726.png), [단계 GIF](/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/methods/sam_half_ellipse/animations/20261022_103429_752726.gif).

`evidence_manifest.json`에는 각 원본의 카메라·프레임 ID·timestamp·원본 SHA-256·마스크 SHA-256·프롬프트·선택된 점 좌표와 해당 카메라별 이미지 경로를 보존했다. `method_comparison.json`에는 방법별 3D proxy와 카메라별 오차가 있다.

## 코드와 실행 범위

`precision_insertion/held_key_tip.py`의 `distal_key_mask_half_ellipse`가 2D 관측 중심을 계산하고, `compare_saved_key_face_methods.py`가 실제 저장 자료 비교·역할별 PNG·GIF를 재현한다. `estimate_saved_key_face_ellipse.py`는 저장된 Qwen·SAM3 결과에서 타원 후보를 만든다. 이 방법은 현재 **읽기 전용 재생으로만 평가**했다. 실제 `retry_session.py`의 `vlm`·`tracked_tip` 경로를 바꾸지 않았고, 로봇 삽입용 XY로 배선하지 않았다. 실시간 적용 전에는 SAM3 마스크와 최소 2개 독립 카메라 결과가 그 경로에서 제시간에 생성되는지 별도 평가가 필요하다.
