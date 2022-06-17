const { response } = require("express");
const express = require("express") // router에서도 express 라는 걸 쓰겠다라는걸 알려줌
const router = express.Router(); // express 안에서 Router 사용하겠다
const conn = require("../config/DB.js")  // 저 경로의 파일안에 있는 내용을 가져온다

module.exports = router;