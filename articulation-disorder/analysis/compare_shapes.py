import numpy as np

def calculate_similarity(coords1, coords2):
    """
    두 프레임의 입 좌표 리스트 [(x1, y1), (x2, y2), ...] 를 받아
    유사도(%)를 계산해서 반환함.
    """
    if len(coords1) != len(coords2):
        return 0.0

    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    # 정규화 (중심을 원점으로 이동하고, 전체 길이를 1로 맞춤)
    def normalize(coords):
        coords -= coords.mean(axis=0)
        norm = np.linalg.norm(coords)
        return coords / norm if norm != 0 else coords

    coords1 = normalize(coords1)
    coords2 = normalize(coords2)

    # 평균 거리 기반 유사도 측정
    distance = np.linalg.norm(coords1 - coords2, axis=1).mean()
    similarity = max(0.0, 100 * (1 - distance))

    return round(similarity, 2)
