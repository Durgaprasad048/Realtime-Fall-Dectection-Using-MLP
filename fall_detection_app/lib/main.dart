import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:url_launcher/url_launcher.dart';

void main() {
  runApp(const FallDetectionApp());
}

class DemoSample {
  final List<double> values;
  final int labelId;
  const DemoSample(this.values, this.labelId);
}

class FallDetectionApp extends StatefulWidget {
  const FallDetectionApp({super.key});

  @override
  State<FallDetectionApp> createState() => _FallDetectionAppState();
}

class _FallDetectionAppState extends State<FallDetectionApp> {
  Interpreter? _interpreter;
  StreamSubscription<AccelerometerEvent>? _accelSub;
  StreamSubscription<GyroscopeEvent>? _gyroSub;
  Timer? _datasetTimer;
  Timer? _imminentFallTimer;

  bool _isModelLoaded = false;
  bool _useDatasetMode = true;

  final List<double> _accBuffer = [];
  final List<double> _gyroBuffer = [];
  final int _windowSize = 20;
  final List<DemoSample> _demoSamples = [];
  int _demoIndex = 0;

  final TextEditingController _caretakerController = TextEditingController(
    text: "+91XXXXXXXXXX",
  );

  static const int _fallClassId = 4;
  static const double _preFallRiskThreshold = 0.85;
  static const double _fallConfidenceThreshold = 0.92;
  static const int _minConsecutiveFalls = 3;
  static const int _minHighRiskWindowsForCountdown = 4;
  static const int _secondsBeforeAutoCall = 5;
  static const int _alertCooldownSeconds = 30;

  int _consecutiveFallWindows = 0;
  int _highRiskWindows = 0;
  bool _sosTriggered = false;
  bool _autoSosEnabled = true;
  bool _autoCallEnabled = true;
  String _predictionText = "Initializing...";
  String _lastStatus = "Monitoring";
  double _fallProbability = 0.0;
  int _secondsToAutoCall = -1;
  DateTime? _lastSosTime;
  DateTime? _lastEmergencyTrigger;
  bool _modelUnavailable = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _stopAllInputs();
    _interpreter?.close();
    _caretakerController.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    try {
      try {
        _interpreter = await Interpreter.fromAsset('assets/model/fall_model.tflite');
      } catch (_) {
        _interpreter = await Interpreter.fromAsset('model/fall_model.tflite');
      }

      if (!mounted) {
        return;
      }

      setState(() {
        _isModelLoaded = true;
        _predictionText = "Model loaded. Starting data stream...";
      });

      await _setInputMode(datasetMode: _useDatasetMode);
    } catch (e) {
      if (!mounted) {
        return;
      }
      debugPrint("Error loading TFLite model: $e");
      setState(() {
        _modelUnavailable = true;
        _predictionText = "Model unavailable on emulator. Running dataset label mode.";
        _lastStatus = "Model load failed, fallback active";
      });
      await _setInputMode(datasetMode: true);
    }
  }

  Future<void> _setInputMode({required bool datasetMode}) async {
    _stopAllInputs();
    _accBuffer.clear();
    _gyroBuffer.clear();
    _demoIndex = 0;

    setState(() {
      _useDatasetMode = datasetMode;
      _lastStatus = datasetMode ? "Using example dataset stream" : "Using live sensors";
    });

    if (datasetMode) {
      await _loadDemoSamples();
      _startDatasetPlayback();
    } else {
      _startSensorStream();
    }
  }

  void _stopAllInputs() {
    _accelSub?.cancel();
    _gyroSub?.cancel();
    _datasetTimer?.cancel();
    _imminentFallTimer?.cancel();
    _accelSub = null;
    _gyroSub = null;
    _datasetTimer = null;
    _imminentFallTimer = null;
  }

  void _startSensorStream() {
    _accelSub = accelerometerEvents.listen((event) {
      _accBuffer.addAll([event.x, event.y, event.z]);
      _checkWindow();
    });

    _gyroSub = gyroscopeEvents.listen((event) {
      _gyroBuffer.addAll([event.x, event.y, event.z]);
      _checkWindow();
    });
  }

  Future<void> _loadDemoSamples() async {
    _demoSamples.clear();
    final assetFiles = [
      'assets/demo/STD_SubjectA_01.csv',
      'assets/demo/WAL_SubjectA_01.csv',
      'assets/demo/FALL_SubjectA_01.csv',
    ];

    for (final path in assetFiles) {
      final raw = await rootBundle.loadString(path);
      final lines = raw.split(RegExp(r'\r?\n')).where((line) => line.trim().isNotEmpty);
      var first = true;
      for (final line in lines) {
        if (first) {
          first = false;
          continue;
        }
        final cols = line.split(',');
        if (cols.length < 6) {
          continue;
        }
        final sample = <double>[];
        for (int i = 0; i < 6; i++) {
          final value = double.tryParse(cols[i].trim());
          if (value == null) {
            sample.clear();
            break;
          }
          sample.add(value);
        }
        if (sample.length == 6 && cols.length >= 7) {
          final labelText = cols[6].trim().toUpperCase();
          final labelId = _labelToClassId(labelText);
          _demoSamples.add(DemoSample(sample, labelId));
        }
      }
    }
  }

  void _startDatasetPlayback() {
    if (_demoSamples.isEmpty) {
      setState(() {
        _predictionText = "No demo samples found.";
        _lastStatus = "Dataset unavailable";
      });
      return;
    }

    _datasetTimer = Timer.periodic(const Duration(milliseconds: 120), (_) {
      final sample = _demoSamples[_demoIndex];
      if (_isModelLoaded) {
        _accBuffer.addAll(sample.values.sublist(0, 3));
        _gyroBuffer.addAll(sample.values.sublist(3, 6));
        _checkWindow();
      } else {
        _runDemoLabelPrediction(sample.labelId);
      }
      _demoIndex = (_demoIndex + 1) % _demoSamples.length;
    });
  }

  void _checkWindow() {
    if (!_isModelLoaded || _interpreter == null) {
      return;
    }

    if (_accBuffer.length >= _windowSize * 3 && _gyroBuffer.length >= _windowSize * 3) {
      final inputWindow = <List<double>>[];
      for (int i = 0; i < _windowSize; i++) {
        inputWindow.add([
          _accBuffer[i * 3],
          _accBuffer[i * 3 + 1],
          _accBuffer[i * 3 + 2],
          _gyroBuffer[i * 3],
          _gyroBuffer[i * 3 + 1],
          _gyroBuffer[i * 3 + 2],
        ]);
      }

      _accBuffer.removeRange(0, _windowSize * 3);
      _gyroBuffer.removeRange(0, _windowSize * 3);
      _runModel(inputWindow);
    }
  }

  void _runModel(List<List<double>> sensorWindow) {
    if (!_isModelLoaded || _interpreter == null) {
      return;
    }

    final input = [sensorWindow];
    final output = [List.filled(7, 0.0)];

    try {
      _interpreter!.run(input, output);
      final probabilities = output[0];
      final predictedClass = probabilities.indexOf(
        probabilities.reduce((a, b) => a > b ? a : b),
      );
      final currentFallProb = probabilities[_fallClassId];

      _handlePrediction(predictedClass, currentFallProb);
    } catch (e) {
      debugPrint("Error running inference: $e");
      setState(() {
        _lastStatus = "Inference failed";
      });
    }
  }

  void _runDemoLabelPrediction(int labelId) {
    final isFall = labelId == _fallClassId;
    final confidence = isFall ? 0.95 : 0.20;
    _handlePrediction(labelId, confidence);
  }

  void _handlePrediction(int predictedClass, double currentFallProb) {
    final highRisk = currentFallProb >= _preFallRiskThreshold;
    if (predictedClass == _fallClassId && currentFallProb >= _fallConfidenceThreshold) {
      _consecutiveFallWindows += 1;
    } else {
      _consecutiveFallWindows = 0;
    }
    if (highRisk) {
      _highRiskWindows += 1;
    } else {
      _highRiskWindows = 0;
    }

    setState(() {
      _fallProbability = currentFallProb;
      if (predictedClass == _fallClassId) {
        _predictionText = "FALL DETECTED (${(currentFallProb * 100).toStringAsFixed(1)}%)";
      } else {
        _predictionText = "Activity: ${_className(predictedClass)}";
      }
    });

    final inCooldown = _isInAlertCooldown();
    if (!_sosTriggered && !inCooldown && highRisk && _highRiskWindows >= _minHighRiskWindowsForCountdown) {
      _startImminentFallCountdown();
    } else if (!highRisk || inCooldown) {
      _cancelImminentFallCountdown();
    }

    if (_consecutiveFallWindows >= _minConsecutiveFalls && !_sosTriggered && !inCooldown) {
      _triggerEmergencySos();
    }
  }

  bool _isInAlertCooldown() {
    if (_lastEmergencyTrigger == null) {
      return false;
    }
    final elapsed = DateTime.now().difference(_lastEmergencyTrigger!);
    return elapsed.inSeconds < _alertCooldownSeconds;
  }

  void _startImminentFallCountdown() {
    if (!_autoCallEnabled || _sosTriggered || _imminentFallTimer != null) {
      return;
    }
    setState(() {
      _secondsToAutoCall = _secondsBeforeAutoCall;
      _predictionText = "About to fall in $_secondsToAutoCall seconds. Calling caretaker...";
      _lastStatus = "High fall risk detected";
    });

    _imminentFallTimer = Timer.periodic(const Duration(seconds: 1), (timer) async {
      if (!mounted || _sosTriggered) {
        timer.cancel();
        _imminentFallTimer = null;
        return;
      }

      if (_fallProbability < _preFallRiskThreshold) {
        _cancelImminentFallCountdown();
        return;
      }

      if (_secondsToAutoCall <= 1) {
        timer.cancel();
        _imminentFallTimer = null;
        await _triggerPreFallEmergencyCall();
        return;
      }

      setState(() {
        _secondsToAutoCall -= 1;
        _predictionText = "About to fall in $_secondsToAutoCall seconds. Calling caretaker...";
      });
    });
  }

  void _cancelImminentFallCountdown() {
    _imminentFallTimer?.cancel();
    _imminentFallTimer = null;
    if (_secondsToAutoCall >= 0 && !_sosTriggered) {
      setState(() {
        _secondsToAutoCall = -1;
        _lastStatus = "Risk normalized";
      });
    }
  }

  Future<void> _triggerPreFallEmergencyCall() async {
    HapticFeedback.mediumImpact();
    setState(() {
      _sosTriggered = true;
      _secondsToAutoCall = -1;
      _lastStatus = "Pre-fall emergency call triggered";
      _predictionText = "High risk of fall. Calling caretaker now...";
      _lastSosTime = DateTime.now();
      _lastEmergencyTrigger = DateTime.now();
    });
    await _callCaretaker();
    if (_autoSosEnabled) {
      await _launchSmsSos();
    }
  }

  Future<void> _triggerEmergencySos() async {
    HapticFeedback.heavyImpact();
    setState(() {
      _sosTriggered = true;
      _lastStatus = "Emergency detected";
      _lastSosTime = DateTime.now();
      _lastEmergencyTrigger = DateTime.now();
    });

    if (_autoSosEnabled) {
      await _launchSmsSos();
    }
  }

  String _buildSosMessage() {
    final now = DateTime.now().toLocal();
    return "SOS: Fall detected at $now. Please check immediately.";
  }

  Future<void> _launchSmsSos() async {
    final number = _caretakerController.text.trim();
    if (number.isEmpty) {
      _setStatus("Caretaker number missing");
      return;
    }
    final encoded = Uri.encodeComponent(_buildSosMessage());
    final smsUri = Uri.parse("sms:$number?body=$encoded");
    final launched = await launchUrl(smsUri, mode: LaunchMode.externalApplication);
    _setStatus(launched ? "SOS message opened" : "Could not open SMS app");
  }

  Future<void> _callCaretaker() async {
    final number = _caretakerController.text.trim();
    if (number.isEmpty) {
      _setStatus("Caretaker number missing");
      return;
    }
    final telUri = Uri.parse("tel:$number");
    final launched = await launchUrl(telUri, mode: LaunchMode.externalApplication);
    _setStatus(launched ? "Calling caretaker" : "Could not start call");
  }

  void _resetEmergency() {
    _cancelImminentFallCountdown();
    setState(() {
      _sosTriggered = false;
      _consecutiveFallWindows = 0;
      _highRiskWindows = 0;
      _secondsToAutoCall = -1;
      _lastStatus = "Monitoring";
    });
  }

  void _setStatus(String value) {
    if (!mounted) {
      return;
    }
    setState(() {
      _lastStatus = value;
    });
  }

  String _className(int id) {
    switch (id) {
      case 0:
        return "Standing";
      case 1:
        return "Walking";
      case 2:
        return "Jogging";
      case 3:
        return "Jumping";
      case 4:
        return "Fall";
      case 5:
        return "Lying";
      case 6:
        return "Running Arm";
      default:
        return "Unknown";
    }
  }

  int _labelToClassId(String label) {
    switch (label) {
      case "STD":
        return 0;
      case "WAL":
        return 1;
      case "JOG":
        return 2;
      case "JUM":
        return 3;
      case "FALL":
        return 4;
      case "LYI":
        return 5;
      case "RA":
        return 6;
      default:
        return 0;
    }
  }

  @override
  Widget build(BuildContext context) {
    final isFall = _predictionText.contains("FALL");
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        backgroundColor: const Color(0xFF0F172A),
        appBar: AppBar(
          title: const Text("Fall Detection + SOS"),
          backgroundColor: const Color(0xFF111827),
        ),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Card(
                color: _sosTriggered ? const Color(0xFF7F1D1D) : const Color(0xFF1F2937),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _predictionText,
                        style: TextStyle(
                          fontSize: 22,
                          color: isFall ? Colors.redAccent : Colors.white,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        "Fall probability: ${(_fallProbability * 100).toStringAsFixed(1)}%",
                        style: const TextStyle(color: Colors.white70, fontSize: 15),
                      ),
                      if (_secondsToAutoCall >= 0) ...[
                        const SizedBox(height: 6),
                        Text(
                          "About to fall in $_secondsToAutoCall seconds. Auto-calling caretaker.",
                          style: const TextStyle(color: Colors.orangeAccent, fontSize: 14),
                        ),
                      ],
                      const SizedBox(height: 6),
                      Text(
                        "Status: $_lastStatus",
                        style: const TextStyle(color: Colors.white70, fontSize: 15),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        "Input mode: ${_useDatasetMode ? "Example datasets" : "Live sensors"}",
                        style: const TextStyle(color: Colors.white70, fontSize: 14),
                      ),
                      if (_modelUnavailable) ...[
                        const SizedBox(height: 6),
                        const Text(
                          "TFLite unavailable on emulator, using dataset label fallback.",
                          style: TextStyle(color: Colors.orangeAccent, fontSize: 13),
                        ),
                      ],
                      if (_lastSosTime != null) ...[
                        const SizedBox(height: 6),
                        Text(
                          "Last SOS: ${_lastSosTime!.toLocal()}",
                          style: const TextStyle(color: Colors.white70, fontSize: 14),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 12),
              Card(
                color: const Color(0xFF1F2937),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        "Caretaker Contact",
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 10),
                      TextField(
                        controller: _caretakerController,
                        style: const TextStyle(color: Colors.white),
                        keyboardType: TextInputType.phone,
                        decoration: const InputDecoration(
                          labelText: "Phone Number",
                          labelStyle: TextStyle(color: Colors.white70),
                          enabledBorder: OutlineInputBorder(
                            borderSide: BorderSide(color: Colors.white38),
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderSide: BorderSide(color: Colors.redAccent),
                          ),
                        ),
                      ),
                      const SizedBox(height: 10),
                      SwitchListTile(
                        value: _autoSosEnabled,
                        onChanged: (value) {
                          setState(() {
                            _autoSosEnabled = value;
                          });
                        },
                        title: const Text(
                          "Auto-SOS on fall",
                          style: TextStyle(color: Colors.white),
                        ),
                        subtitle: const Text(
                          "Opens SMS app automatically when fall is confirmed",
                          style: TextStyle(color: Colors.white70),
                        ),
                        activeColor: Colors.redAccent,
                      ),
                      SwitchListTile(
                        value: _autoCallEnabled,
                        onChanged: (value) {
                          setState(() {
                            _autoCallEnabled = value;
                          });
                          if (!value) {
                            _cancelImminentFallCountdown();
                          }
                        },
                        title: const Text(
                          "Auto-call on high fall risk",
                          style: TextStyle(color: Colors.white),
                        ),
                        subtitle: const Text(
                          "Warns with countdown and calls caretaker automatically",
                          style: TextStyle(color: Colors.white70),
                        ),
                        activeColor: Colors.orangeAccent,
                      ),
                      SwitchListTile(
                        value: _useDatasetMode,
                        onChanged: _isModelLoaded
                            ? (value) {
                                _setInputMode(datasetMode: value);
                              }
                            : null,
                        title: const Text(
                          "Use example dataset stream",
                          style: TextStyle(color: Colors.white),
                        ),
                        subtitle: const Text(
                          "Disable this later to use live phone sensors",
                          style: TextStyle(color: Colors.white70),
                        ),
                        activeColor: Colors.tealAccent,
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 12),
              Wrap(
                spacing: 10,
                runSpacing: 10,
                children: [
                  ElevatedButton.icon(
                    onPressed: _launchSmsSos,
                    icon: const Icon(Icons.sos),
                    label: const Text("Send SOS Now"),
                    style: ElevatedButton.styleFrom(backgroundColor: Colors.redAccent),
                  ),
                  ElevatedButton.icon(
                    onPressed: _callCaretaker,
                    icon: const Icon(Icons.call),
                    label: const Text("Call Caretaker"),
                  ),
                  OutlinedButton.icon(
                    onPressed: _resetEmergency,
                    icon: const Icon(Icons.restart_alt),
                    label: const Text("Reset Alert"),
                    style: OutlinedButton.styleFrom(foregroundColor: Colors.white),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
