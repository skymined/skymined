#coding #cg

3Point 논문에 인볼브 되기 전 두 가지 일을 했다.

첫 번째는 Forward Kinematic 실습이구요, 두 번째는 Unity를 이용한 게임 만들기.

### **Forward Kinematics**

일단 FK는 관절의 회전값이 주어졌을 때, 각 관절과 end-effector가 공간상의 어디에 위치해야 하는지를 계산하는 과정이다. 우리는 각 관절이 부모 관절 기준(local)으로 얼마나 회전했는지는 알고 있지만, 그 관절이 세계(gloabl) 좌표계에서 어디에 있는지는 알지 못한다. 그래서 FK는 이 로컬 정보를 뼈 구조(hierarchy)를 따라서 누적하면서 각 관절의 전역 위치와 전역 회전을 계산하는 절차다. 

### 주어진 데이터

이번 실습에서는 완성된 포즈나 관절 위치 뿐만 아니라 조합해서 이를 만들어내는 것이었다.

총 8개의 데이터셋이 있고 각각에는 다양한 형태의 데이터가 담겨져 있었다.

하나씩 하면서 알아보는 것으로.

---

### 1번째

> joint hierarhcy + joint global position

```python
# 모션 데이터 로딩, 데이터 shape 확인
    data = np.load(f'./motion_data/motion_{MOTION_IDX}.npz')
    for item in data:
        print(f'{item:<35}shape: {data[item].shape}')
```

해당 방식으로 데이터를 뽑아보았다.

```python
joint_hierarchy                    shape: (52,)
joint_global_position              shape: (2751, 52, 3)
```

\`joint\_hierarchy\`는 캐릭터 스켈레톤의 계층 구조(부모-자식 관계)를 나타낸다. 그러니까 이 데이터셋에는 52개의 관절이 있다는 뜻

만약 joint\_hierarchy\[5\]=2라고 한다면, 5번 관절의 부모는 2번 관절이라는 뜻.

\`joint\_global\_position\`은 각 프레임마다 모든 관절이 3D 공간상에서 어디에 위치하는지 절대 좌표를 담고 있는 데이터.

(프레임 수, 관절의 개수, 각 관절의 좌표값)이므로 현재 이 데이터는 2751장의 장면, 52개의 관절, 그리고 각 관절은 x,y,z 좌표값을 가졌다는 뜻.
![[img.gif]]

짜잔. 신기하네요.

### 2번째

> joint hierarhcy + root position + T-pose joint global position + joint local rotation(euler angle)  
> (euler angle은 zxy 순서이며, radian이 아닌 degree)

여기서부터는 이제 계산을 해야 합니다. 

![[Pasted image 20260228181554.png]]

#### 데이터 뜯어보기

일단 구조가 어떻게 되어 있는지 모르니 하나하나 뜯어봤습니다.

먼저 상위 구조부터 보자.

```python
a = data['joint_hierarchy']
print(a.shape)
print(a)

>>>
(52,)
[-1  0  0  0  1  2  3  4  5  6  7  8  9  9  9 12 13 14 16 17 18 19 20 22
 23 20 25 26 20 28 29 20 31 32 20 34 35 21 37 38 21 40 41 21 43 44 21 46
 47 21 49 50]
```

52개의 관절들의 부모 관절 인덱스를 표현한 것. 말하자면 `parent[j]=-1` 과 같은 형태.

그 다음은 `joint_tpose_global_position` 차례.

```python
>>>

(52, 3)
[[-2.61746347e-03 -2.35446870e-01  1.64059848e-02]
 [ 5.21157458e-02 -3.18507820e-01 -4.99322079e-03]
 [-5.89251518e-02 -3.26954424e-01 -1.34882331e-03]
 [ 1.95406377e-03 -1.11230776e-01 -1.55844837e-02]
 [ 9.97703150e-02 -7.22852111e-01  6.44239224e-03]
 [-1.06916338e-01 -7.24975169e-01 -4.66445461e-03]
 [ 6.77851588e-03  3.08516026e-02  1.44082215e-02]
 [ 8.61195251e-02 -1.16153193e+00 -2.87204366e-02]
 (...)
```

각 관절이 일반 상태일 때의 global 좌표를 의미하는 것 같다.

`root_position`

```python
>>>
(4839, 3)
[[1.13945566 1.63128461 0.98450536]
 [1.13965111 1.63278147 0.98435836]
 [1.13953751 1.63411354 0.9842399 ]
 ...
 [0.82484759 1.00675018 0.96363256]
 [0.82705304 1.00363746 0.96456952]
 [0.82919194 1.00078028 0.96570393]]
```

parent\[j\]=-1 이었던 관절이 4839 각 프레임마다 어느 좌표에 있었는지를 나타내는 것

마지막으로 `joint_local_rotaton`

```python
>>>
(4839, 52, 3)
[[[-102.03730645   61.15115905  -97.67397728]
  [   4.89989086  -11.41543582    6.45708839]
  [   3.61069989  -17.88493525   -0.9230529 ]
```

4839 프레임 동안 52개의 관절의 회전. eular angle을 나타내며 radian이 아닌 degree라는 사실  
로컬이라는 뜻은 부모 관절 좌표계 기준으로 자식 관절이 얼마나 회전했는지를 의미한다고 한다.

#### 우선 하나의 모션만 구해보자

프레임별로 생각하면 머리가 터질 것 같아서 일단 다음 프레임의 모션을 구해보기로 했다.

```python
joint_tpose_global_position[0]
= [-0.0026, -0.2354, 0.0164]
```

T-pose에서 root의 관절이 원점이 아니라 어떤 위치에 서 있다는 뜻이다.  
반면 `root_positioni[t]`는 매 프레임에서 root의 전역 위치를 따로 주고 있는데 이 둘을 그대로 동시에 쓰면 root의 위치가 이중으로 적용될 위험이 생긴다고 한다. FK의 핵심원칙 -> T-pose는 '뼈의 상대 구조'만 제공해야 하고, 실제 공간상의 위치(transition)은 오직 root\_position이 담당해야 한다.

offset을 만들 때는 root의 위치가 사라지긴 하지만 회전의 기준점에서는 사라지지 않기 때문에 FK 하기 전에 T-pose를 root 기준으로 정렬해야 한다는 것 같음.

그러므로 T-pose 전체를 root 기준으로 한 번 정렬한다.

```python
root_tpose = joint_tpose_global_position[0]
```

이제 rotation 값을 구해야 하는데.....

```python
joint_local_rotation[0,0]
= [-102.03730645   61.15115905  -97.67397728] # [z,x,y]
```

-   요 말은 z 축으로 -102도, x축으로 61도, y축으로 -97도 회전했다는 뜻
-   이 3번의 회전을 하나의 회전 행렬 R로 합쳐야 한다.

```python
R_local = Rz(z) @ Rx(x) @ Ry(y)
```

그래서 회전 행렬 값을 return 해주는 함수를 만들었다.

```python
# 축별로 회전 행렬을 3개 만들어보자
def euler_zxy_deg_to_R(zxy_deg):
    # degree -%3E radian으로 바꾸기
    z, x, y = np.deg2rad(zxy_deg)
    # 각 축의 회전 행렬 값
    cz, sz = np.cos(z), np.sin(z)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [ 0,   0,  1]
])

    Rx = np.array([
        [1,  0,   0 ],
        [0, cx, -sx],
        [0, sx,  cx]
    ])

    Ry = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy]
    ])

    # return으로 합치기
    return Ry @ Rx @ Rz)
```

그리고 실제 offset을 바로 돌릴 수 있는 회전 연산자 만들어 두기

![[Pasted image 20260228181608.png]]

이것은 내가 마지막에 Rz @ Rx @ Ry 라고 해서 나온 결과물. 

내가 이 결과물을 보고 이건 대체 어떻게 디버깅을 하냐고 물어봤는데, 눈으로 보고 확인해야 한다는 답을 얻었다.

진짜 말이 안된다. 모션 캡쳐를 하는 사람들은 정말 대단한 사람들이다.

어느 부분이 문제인지 print를 찍어봤으나 숫자만 나오는...

```python
joint_local_rotation_matrix = np.zeros((52,3,3))
    for j in range(52):
        joint_local_rotation_matrix[j] = euler_zxy_deg_to_R(joint_local_rotation[0,j])
```

지금 내가 가진 것을 정리해보자

-   `joint_offset`: T-pose에서 부모 -> 자식 뼈 벡터 (root 기준으로 정렬함!)
-   `joint_local_rotation_matrix`: 각 관절의 부모 기준 로컬 회전 행렬! (그래서 52, 3, 3) -> 프레임 하나만!

그런데 지금 내가 가지고자 하는 것은 `joint_global_position`, `joint_global_rotation_matrix`임.

FK 누적 규칙에 따라

1.  부모 전역 회전에 자식 로컬 회전을 붙이고
2.  부모 위치에서, 부모 전역 회전으로 offset을 돌려 더하고
3.  root만 시작점이라 별도 
4.  `joint_global_rotation_matrix = np.zeros((52, 3, 3)) joint_global_position = np.zeros((52, 3)) joint_global_rotation_matrix[0] = joint_local_rotation_matrix[0] joint_global_position[0] = root_position[0]`

```python
for j in range(52):
parent = joint_hierarchy[j]
if parent == -1:
continue
# 전역 회전 = 부모 전역 회전 @ 내 로컬 회전
joint_global_rotation_matrix[j] = joint_global_rotation_matrix[parent] @ joint_local_rotation_matrix[j]
# 전역 위치 = 부모 전역 위치 + 부모 전역 회전 @ 내 오프셋
joint_global_position[j] = joint_global_position[parent] + joint_global_rotation_matrix[j] @ joint_offset[j]
```

````python
offset 오차 있는지 확인
```python
len_offset = np.linalg.norm(joint_offset[j])
len_world  = np.linalg.norm(joint_global_position[j] - joint_global_position[p])
print("offset length:", len_offset)
print("world  length:", len_world)
print("difference:", abs(len_offset - len_world))
````

![[Pasted image 20260228181627.png]]

결과물이 나왔다!

굉장히 기뻐요~

나머지 3번부터 8번까지는 나중에 해보겠습니다...