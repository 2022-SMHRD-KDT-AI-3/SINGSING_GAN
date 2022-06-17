const express = require ("express");
const app = express();
const router = require("./router/router.js"); // 라우터.js 파일을 변수에 담아 사용
const bodyparser = require("body-parser"); // post방식을 사용하기 위해 작성 
const ejs = require("ejs");
const session = require("express-session"); // 세션기능을 사용하기 위한 모듈
const session_mysql_save = require("express-mysql-session"); // 세션기능을 저장하기 위한 모듈

let s_m_s = new session_mysql_save(DB_info); // DB_info의 정보들을 'session_mysql_save'에 저장

app.use(express.static("./public"));  // 현재 프로젝트에 정적파일 폴더지정

// 2번째 미들웨어 역할
app.set("view engine", "ejs"); // express 내부적으로 engine이 설정되어있기 때문에 set기능 사용
app.use(session({
    secret : "smart",  // 비밀
    resave : false,    // 세션값을 저장할때 새롭게 저장할
    saveUninitialized : true,   // 세션값을 저장할껀지 안할껀지
    store : s_m_s    // 어디에 저장할껀지
}));


app.use(bodyparser.urlencoded({extended:false})); // ????를 사용하지 않겠다라고 지정
app.use(router);
app.listen(3000);