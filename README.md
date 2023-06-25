# ai
AI of clothing recommendation service tailored to temperature and personal schedule

## 📚 API
<table>
  <tr>
    <th>num</th>
    <th>name</th>
    <th>method</th>
    <th>uri</th>
  </tr>
  <tr>
    <td>1</td>
    <td>일정과 지역 분류</td>
    <td>GET</td>
    <td>/api/schedules</td>
  </tr>
  <tr>
    <td>2</td>
    <td>옷 추천</td>
    <td>GET</td>
    <td>/api/clothes</td>
  </tr>
</table>

### example
<table>
  <tr>
    <th>num</th>
    <th>request</th>
    <th>response</th>
  </tr>
  <tr>
    <td>1</td>
    <td>?title=응주랑 런닝&amp;region=뚝섬</td>
    <td>{ "plan": "운동", "region": "광진구" }</td>
  </tr>
  <tr>
    <td>2</td>
    <td>
      ?gender=1&amp;humidity=5&amp;wind_speed=3 <br/>
      &amp;rain=0&amp;temp=-14&amp;schedule=데이트
    </td>
    <td>{ "bottom": 158, "top": 108 }</td>
  </tr>
</table>

## 💡 서비스 구축
### 데이터 수집
![image](https://github.com/todayclothes/ai/assets/87798704/40e63f20-cf37-4554-8e98-a2ac95be2bfd)

### 위치 및 일정 필터링 모델
![image](https://github.com/todayclothes/ai/assets/87798704/0a2055a4-a3e8-4035-b0bb-7b48bbe39a73)

### 옷 추천 모델
![image](https://github.com/todayclothes/ai/assets/87798704/e602e949-5c98-4315-9ba9-1143dc85e1be)

### 라이브러리 및 모델 선정 과정
![image](https://github.com/todayclothes/ai/assets/87798704/7c86d2d0-381c-4df3-9bd0-7d13b25f538c)

## 👨‍💻 created by
<table>
  <tr>
    <td>
      <img src="https://avatars.githubusercontent.com/ByeongmokKim" width=150 />
    </td>
    <td>
      <img src="https://avatars.githubusercontent.com/reeruryu" width=150 />
    </td>
  </tr>
  <tr>
    <td align=center>
      <a href="https://github.com/ByeongmokKim">@ByeongmokKim</a>
    </td>
    <td align=center>
      <a href="https://github.com/reeruryu">@reeruryu</a>
    </td>
  </tr>
</table>
