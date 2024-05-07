  import 'dart:async';

import 'package:flutter/material.dart';

import 'package:camera/camera.dart';
import './pose_detector_view.dart';


import 'dart:typed_data';
import 'package:flutter/material.dart';


// 카메라 목록 변수
List<CameraDescription> cameras = [];

Future<void> main() async {
  // 비동기 메서드를 사용함
  WidgetsFlutterBinding.ensureInitialized();
  // 사용 가능한 카메라 목록 받아옴
  cameras = await availableCameras();
  // 앱 실행
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: ShowFairytale(),
    );
  }
}



  class ShowFairytale extends StatelessWidget {
    ShowFairytale({super.key});

    @override
    Widget build(BuildContext context) {
      return Scaffold(
        body: Container(
          color: Color.fromARGB(0xFF, 0xB9, 0xEE, 0xFF),
          child: Column(
            children: [
              Container(
                alignment: Alignment.topLeft,
                margin: EdgeInsets.fromLTRB(20, 20, 0, 0),
                child: Text(
                  '동화 만들기',
                  style: TextStyle(
                    fontSize: 60,
                    color: Color.fromARGB(0xFF, 0x3B, 0x2F, 0xCA),
                  ),
                ),
              ),
              Expanded(
                child: Container(
                  width: double.infinity,
                  height: double.infinity,
                  color: Color(0xFFFFFFFF),
                  margin: EdgeInsets.all(25),
                  child: Story(),
                ),
              ),
            ],
          ),
        ),
      );
    }
  }

  class Story extends StatefulWidget {
    Story({super.key});

    var images;

    @override
    State<Story> createState() => _StoryState();
  }

  class _StoryState extends State<Story> {
    int index = 0;
    int max = 3;

    @override
    Widget build(BuildContext context) {
      var images = widget.images;

      Future.delayed(
        const Duration(milliseconds: 2000),
            () {
          if (index < max-1) {
            setState(
                  () {
                index += 1;
              },
            );
          }
        },
      );

      return Container(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Flexible(
              child: Container(
                color: Colors.white,
                child: Stack(
                  children: [
                    Positioned(
                      left: 0,
                      right: 0,
                      child: Container(
                        width: 500,
                        height: 500,
                        child: Image.asset("assets/a.png"),
                      ),
                    ),
                    Positioned(
                      left: 0,
                      right: 0,
                      child: PoseDetectorView(),
                    )
                  ],
                ),
              ),
            ),
            Flexible(
              child: Container(
                width: double.infinity,
                height: double.infinity,
                child: Column(
                  children: [
                    Flexible(
                      flex: 3,
                      child: Container(
                        height: double.infinity,
                        width: double.infinity,
                        alignment: Alignment.center,
                        child: Text(
                          "asd",
//                        gSceneModel.scriptModelList[index].scene_contents,
                          style: TextStyle(fontSize: 40),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      );
    }
  }
