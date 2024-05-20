import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:frontend/pose_detector_view.dart';
import 'package:frontend/constants/dummy_data.dart';
import 'package:frontend/constants/action_list.dart';

class ShowFairytale extends StatelessWidget {
  ShowFairytale({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        color: Color.fromARGB(0xFF, 0xC9, 0xEE, 0xFF),
        child: Column(
          children: [
            Container(
              alignment: Alignment.topLeft,
              margin: EdgeInsets.fromLTRB(20, 20, 0, 0),
              child: Text(
                '동화 만들기',
                style: TextStyle(
                  fontSize: 60,
                  fontWeight: FontWeight.w600,
                  color: Color.fromARGB(0xFF, 0x13, 0x13, 0x13),
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

  @override
  State<Story> createState() => _StoryState();
}

class _StoryState extends State<Story> {
  //int index = 0;
  int max = 8;

  @override
  Widget build(BuildContext context) {
    ValueNotifier(clr_index);
    return ValueListenableBuilder<int>(
      valueListenable: clr_index,
      builder: (context, value, _) {
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
                        top: 0,
                        child: Container(
                          width: 500,
                          height: 500,
                          // child: Image.memory(
                          //   base64Decode(gSceneModel!.b64_images
                          //       .elementAt(clr_index.value)),
                          //   height: 500,
                          //   width: 500,
                          // ),
                          child: Image.asset(imgs.elementAt(clr_index.value)),
                        ),
                      ),
                      Positioned(
                        left: -200,
                        top: -260,
                        child: Transform.scale(
                          scale: 1,
                          child: Container(
                            child: PoseDetectorView(),
                          ),
                        ),
                      ),
                      // Positioned(
                      //   bottom: 0,
                      //   left: 0,
                      //   child: Camera(),
                      // )
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
                          child: Container(),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
