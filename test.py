   <div class="result">
        {% if predictions %}
        <h2>ผลการวิเคราห์!:</h2>
        <ul class="predictions">
            {% for prediction in predictions %}
            <li>{{ prediction }}</li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>





 font-family: Arial, sans-serif;
    max-width: 100%;
    margin: 0 auto;
    padding: 20px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    background-color: #f2f2f2;
    display: flex;
    flex-direction: column;
    align-items: center; /* จัดให้อยู่ตรงกลางแนวนอน */
    justify-content: center; /* จัดให้อยู่ตรงกลางแนวตั้ง */
    min-height: 100vh; /* ความสูงขั้นต่ำของหน้าเว็บ */




 font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            background-color: #f2f2f2;
            /* กำหนดพื้นหลังเป็นรูปภาพ */
            background-image: url("/static/image/background.jpeg");
            background-size: cover;
            background-position: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh; /* ความสูงขั้นต่ำของหน้าเว็บ */
            color: white; /* กำหนดสีของข้อความเป็นสีขาว */
            font-family: Arial, sans-serif;












<!DOCTYPE html>
<html>
<head>
    <title>ยินดีต้อนรับเข้าสู่ระบบตรวจวัดความชื้นเมล็ดข้าวเปลือก</title>
    <style>
        /* สไตล์ของ CSS ที่ควบคุมหน้าเว็บ */
        body {       
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            background-color: #f2f2f2;
            /* กำหนดพื้นหลังเป็นรูปภาพ */
            background-image: url("/static/image/background.jpeg");
            background-size: cover;
            background-position: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh; /* ความสูงขั้นต่ำของหน้าเว็บ */
            color: white; /* กำหนดสีของข้อความเป็นสีขาว */
        }

        h1 {
            text-align: center;
            color: white; /* กำหนดสีของข้อความเป็นสีขาว */
            font-size: 36px; /* กำหนดขนาดตัวอักษรให้ใหญ่ขึ้น */
            font-weight: bold; /* กำหนดความหนาของตัวอักษร */
            text-transform: uppercase; /* กำหนดให้ตัวอักษรเป็นตัวใหญ่ทั้งหมด */
            margin-top: 50px; /* เพิ่ม margin ด้านบนเพื่อให้เนื้อหาอยู่กึ่งกลางของหน้าเว็บ */
        }

        p {
            font-size: 24px; /* กำหนดขนาดตัวอักษรให้ใหญ่ขึ้น */
            text-align: center;
            margin-bottom: 30px; /* เพิ่ม margin ด้านล่างเพื่อให้เนื้อหาอยู่กึ่งกลางของหน้าเว็บ */
        }

        .start-button {
            display: flex;
            margin: 20px auto;
            padding: 10px 20px;
            background-color: white; /* กำหนดสีพื้นหลังของปุ่มเป็นสีขาว */
            color: #6d88ba; /* กำหนดสีของตัวอักษรในปุ่มเป็นสีน้ำเงิน */
            border: none;
            border-radius: 4px;
            font-size: 16px;
            font-weight: bold; /* กำหนดความหนาของตัวอักษร */
            cursor: pointer;
            text-decoration: none;
        }

        .start-button:hover {
            background-color: #e1eaaf;
        }

        .app-image {
            display: flex;
            margin: 20px auto;
            max-width: 100%;
        }
        .center {
        text-align: center;
        }
    </style>
</head>
<body>
    {% include 'navbar.html' %} <!-- เรียกใช้ Navbar -->
    <main>
        <h1>ยินดีตีอนรับเข้าสู่ระบบตรวจวัดความชื้นเมล็ดข้าวเปลือก</h1>
        <p>คำแนะนำ: ไม่ควรนำรูปภาพหรือรูปถ่ายอื่นๆที่ไม่ใช่เมล็ดข้าวเปลือกเข้าสู่ระบบไม่เช่นนั้นอาจทำให้ผลลัพธ์มีความผิดพลาด</p>
        <a href="/index" class="start-button">เข้าสู่การวิเคราะห์</a>
    </main>
</body>
</html>




{% include 'navbar.html' %} <!-- เรียกใช้ Navbar -->