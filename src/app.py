import sys
import os
import shutil
import subprocess
import html
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout,
                              QHBoxLayout, QTextBrowser, QLineEdit, QPushButton, QLabel,
                              QFrame, QMessageBox, QFileDialog, QSizePolicy, QComboBox)
from PyQt6.QtCore import Qt, QThread, QUrl, QBuffer, QIODevice
from PyQt6.QtGui import QFont, QColor, QPalette, QImage
from pipeline_worker import (RecognitionWorker, PlacementWorker, ModifySceneWorker, IntentWorker,
                              ImageSceneWorker, ValidationWorker, SRC_DIR, SCENE_OUTPUT_FILE,
                              objets_list, OBJETS_DIR)
from ui.scene_3d_view import SceneView3D
from conversation_session import ConversationSession
print("[import] app OK")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAI2D Real-to-Sim")
        self.setMinimumSize(1100, 700)
        self._current_objetsList = None
        self._thread = None
        self._worker = None
        self._session = ConversationSession()
        self._pending_intent = None
        self._pending_instruction = None
        self._reference_image_path = None
        self._current_theme = "dark"

        self._setup_ui()
        self._dark_theme()
        self._append_message("Decrivez une scene ou chargez une image.", "system")

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        title = QLabel("PAI2D Real-to-Sim")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("", 15, QFont.Weight.Bold))
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header_row.addWidget(title)

        self.theme_btn = QPushButton("🌸")
        self.theme_btn.setObjectName("theme_btn")
        self.theme_btn.setFixedSize(34, 34)
        self.theme_btn.setToolTip("Changer le theme")
        self.theme_btn.clicked.connect(self._toggle_theme)
        header_row.addWidget(self.theme_btn)
        root_layout.addLayout(header_row)

        # Barre de selection de version
        version_bar = QFrame()
        version_bar.setObjectName("version_bar")
        version_bar.setFrameShape(QFrame.Shape.NoFrame)
        version_bar.setFixedHeight(34)
        vb_layout = QHBoxLayout(version_bar)
        vb_layout.setContentsMargins(4, 0, 4, 0)
        vb_layout.setSpacing(6)

        vb_layout.addStretch()

        txt_lbl = QLabel("Texte")
        txt_lbl.setObjectName("version_label")
        vb_layout.addWidget(txt_lbl)

        self.text_version_combo = QComboBox()
        self.text_version_combo.setObjectName("version_combo")
        self.text_version_combo.setFixedWidth(190)
        for key, label in [
            ("V1.1.1", "V1.1.1 — phi3 fine-tune"),
            ("V1.1",   "V1.1 — embeddings"),
            ("V1",     "V1 — LLM"),
            ("V2",     "V2 — LLM only"),
        ]:
            self.text_version_combo.addItem(label, key)
        self.text_version_combo.setCurrentIndex(0)
        self.text_version_combo.currentIndexChanged.connect(self._on_text_version_changed)
        vb_layout.addWidget(self.text_version_combo)

        vb_layout.addSpacing(18)

        img_lbl = QLabel("Image")
        img_lbl.setObjectName("version_label")
        vb_layout.addWidget(img_lbl)

        self.image_version_combo = QComboBox()
        self.image_version_combo.setObjectName("version_combo")
        self.image_version_combo.setFixedWidth(160)
        self.image_version_combo.addItem("V3 — vision LLM", "V3")
        self.image_version_combo.currentIndexChanged.connect(self._on_image_version_changed)
        vb_layout.addWidget(self.image_version_combo)

        vb_layout.addStretch()

        root_layout.addWidget(version_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.addWidget(self._build_chat_panel())
        splitter.addWidget(self._build_scene_panel())
        splitter.setSizes([420, 680])
        root_layout.addWidget(splitter, stretch=1)

    def _build_chat_panel(self):
        panel = QFrame()
        panel.setObjectName("chat_panel")
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        label = QLabel("Chat")
        label.setFont(QFont("", 12, QFont.Weight.Bold))
        label.setObjectName("panel_title")
        top_row.addWidget(label)
        top_row.addStretch()

        self.new_thread_btn = QPushButton("+ Nouveau thread")
        self.new_thread_btn.setObjectName("new_thread_btn")
        self.new_thread_btn.setFixedHeight(28)
        self.new_thread_btn.clicked.connect(self._on_new_thread)
        top_row.addWidget(self.new_thread_btn)
        layout.addLayout(top_row)

        self.chat_display = QTextBrowser()
        self.chat_display.setReadOnly(True)
        self.chat_display.setOpenLinks(False)
        self.chat_display.anchorClicked.connect(self._on_link_clicked)
        layout.addWidget(self.chat_display, stretch=1)

        self.status_label = QLabel()
        self.status_label.setTextFormat(Qt.TextFormat.RichText)
        self.status_label.setObjectName("status_label")
        layout.addWidget(self.status_label)

        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Decrivez votre scene...")
        self.prompt_input.returnPressed.connect(self._on_send)
        input_row.addWidget(self.prompt_input, stretch=1)

        self.image_btn = QPushButton("📎")
        self.image_btn.setObjectName("image_btn")
        self.image_btn.setFixedWidth(55)
        self.image_btn.setToolTip("Generer la scene depuis une image (V3)")
        self.image_btn.clicked.connect(self._on_load_image)
        input_row.addWidget(self.image_btn)

        self.send_btn = QPushButton("Envoyer")
        self.send_btn.setObjectName("send_btn")
        self.send_btn.setFixedWidth(90)
        self.send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self.send_btn)

        layout.addLayout(input_row)
        return panel

    def _build_scene_panel(self):
        panel = QFrame()
        panel.setObjectName("scene_panel")
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        label = QLabel("Apercu 3D")
        label.setFont(QFont("", 12, QFont.Weight.Bold))
        label.setObjectName("panel_title")
        layout.addWidget(label)

        self.scene_view = SceneView3D()
        layout.addWidget(self.scene_view, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.download_btn = QPushButton("Telecharger")
        self.download_btn.setObjectName("download_btn")
        self.download_btn.setEnabled(False)
        self.download_btn.setFixedHeight(38)
        self.download_btn.setFont(QFont("", 10, QFont.Weight.DemiBold))
        self.download_btn.clicked.connect(self._on_download_scene)
        btn_row.addWidget(self.download_btn)

        self.genesis_btn = QPushButton("Simulation Genesis")
        self.genesis_btn.setObjectName("genesis_btn")
        self.genesis_btn.setEnabled(False)
        self.genesis_btn.setFixedHeight(38)
        self.genesis_btn.setFont(QFont("", 10, QFont.Weight.DemiBold))
        self.genesis_btn.clicked.connect(self._on_launch_genesis)
        btn_row.addWidget(self.genesis_btn)

        self.validate_btn = QPushButton("Valider la scene")
        self.validate_btn.setObjectName("validate_btn")
        self.validate_btn.setEnabled(False)
        self.validate_btn.setFixedHeight(38)
        self.validate_btn.setFont(QFont("", 10, QFont.Weight.DemiBold))
        self.validate_btn.clicked.connect(self._on_validate_scene)
        btn_row.addWidget(self.validate_btn)

        layout.addLayout(btn_row)
        return panel

    # ------------------------------------------------------------------ #
    #                         AFFICHAGE DU CHAT                           #
    # ------------------------------------------------------------------ #

    def _append_message(self, text, role):
        is_pink = self._current_theme == "pink"

        if role == "system":
            color = "#b08090" if is_pink else "#6272a4"
            html_msg = (
                f'<div style="text-align:center; margin:6px 0;">'
                f'<span style="font-size:11px; color:{color}; font-style:italic;">{text}</span>'
                f'</div>'
            )

        elif role == "error":
            bg = "#fde8e8" if is_pink else "#2d1010"
            fg = "#9b1c1c" if is_pink else "#fca5a5"
            html_msg = (
                f'<table width="100%" cellspacing="0" cellpadding="0" style="margin:4px 0;">'
                f'<tr><td bgcolor="{bg}" style="padding:9px 13px;">'
                f'<span style="color:{fg}; font-size:13px;">&#9888; {text}</span>'
                f'</td></tr></table>'
            )

        elif role == "user":
            if is_pink:
                sender_c, bg, fg = "#9b3050", "#fce4ea", "#3d1a26"
            else:
                sender_c, bg, fg = "#a78bfa", "#2e1f5e", "#ddd6fe"
            html_msg = (
                f'<table width="100%" cellspacing="0" cellpadding="0" style="margin:5px 0;">'
                f'<tr><td width="20%"></td>'
                f'<td bgcolor="{bg}" style="padding:10px 14px;">'
                f'<div style="font-size:13px; color:{fg}; text-align:left;">{text}</div>'
                f'</td></tr></table>'
            )

        else:  # assistant
            if is_pink:
                sender_c, bg, fg = "#c06070", "#fff0f3", "#3d1a26"
            else:
                sender_c, bg, fg = "#7dd3fc", "#182130", "#c8dff7"
            html_msg = (
                f'<table width="100%" cellspacing="0" cellpadding="0" style="margin:5px 0;">'
                f'<tr><td bgcolor="{bg}" style="padding:10px 14px;">'
                f'<div style="color:{sender_c}; font-size:9px; font-weight:bold; '
                f'letter-spacing:1px; margin-bottom:4px;">ASSISTANT</div>'
                f'<div style="font-size:13px; color:{fg};">{text}</div>'
                f'</td><td width="20%"></td></tr></table>'
            )

        self.chat_display.append(html_msg)
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _append_images(self, paths):
        """Affiche les thumbnails des images selectionnees dans le chat."""
        is_pink = self._current_theme == "pink"
        sender_c = "#9d174d" if is_pink else "#a78bfa"

        imgs_html = ""
        for path in paths:
            img = QImage(path)
            if img.isNull():
                continue
            if img.width() > 220:
                img = img.scaledToWidth(220, Qt.TransformationMode.SmoothTransformation)
            buf = QBuffer()
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            img.save(buf, "PNG")
            b64 = bytes(buf.data().toBase64()).decode()
            buf.close()
            imgs_html += (
                f'<img src="data:image/png;base64,{b64}" '
                f'style="border-radius:10px; margin:2px 0; display:inline-block;" />'
            )

        if not imgs_html:
            return

        html_msg = (
            f'<div style="text-align:right; margin:8px 6px 6px 6px;">'
            f'{imgs_html}</div>'
        )
        self.chat_display.append(html_msg)
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------ #
    #                     GESTION DES WORKERS (QThread)                   #
    # ------------------------------------------------------------------ #

    def _launch_worker(self, worker):
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "En cours", "Generation en cours, veuillez patienter.")
            return False

        self.send_btn.setEnabled(False)
        self._update_status("En cours...", "working")

        self._thread = QThread()
        self._worker = worker
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.finished.connect(self._thread.deleteLater)

        if hasattr(self._worker, 'status_update'):
            self._worker.status_update.connect(self._on_status_update)

        return True

    def _start_thread(self):
        self._thread.start()

    # ------------------------------------------------------------------ #
    #                         ENVOI DU PROMPT                             #
    # ------------------------------------------------------------------ #

    def _on_send(self):
        instruction = self.prompt_input.text().strip()
        if not instruction:
            return

        if self._session.pending_unrecognized:
            self._append_message("Veuillez d'abord resoudre les objets non reconnus ci-dessus.", "system")
            return

        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "En cours", "Generation en cours, veuillez patienter.")
            return

        self._append_message(html.escape(instruction), "user")
        self._session.add_message("user", instruction)
        self.prompt_input.clear()
        self.download_btn.setEnabled(False)
        self.genesis_btn.setEnabled(False)
        self.validate_btn.setEnabled(False)

        if self._session.objets_list is None:
            print(f"[PIPELINE] Pas de scene existante -> nouvelle scene | prompt: '{instruction}'")
            self._launch_recognition(instruction)
        else:
            print(f"[PIPELINE] Scene existante -> classification de l'intent | prompt: '{instruction}'")
            self._pending_instruction = instruction
            worker = IntentWorker(instruction, self._session.scene_description())
            if not self._launch_worker(worker):
                return
            worker.intent_classified.connect(self._on_intent_classified)
            self._start_thread()

    def _on_intent_classified(self, intent):
        print(f"[INTENT] Classifie : '{intent}'")
        self._pending_intent = intent

    # ------------------------------------------------------------------ #
    #              LANCEMENT DES DIFFERENTS WORKERS                       #
    # ------------------------------------------------------------------ #

    def _launch_recognition(self, prompt):
        print(f"[WORKER] Lancement RecognitionWorker | prompt: '{prompt}'")
        self._reference_image_path = None
        self._session.original_prompt = prompt
        worker = RecognitionWorker(prompt)
        if not self._launch_worker(worker):
            return
        worker.recognition_done.connect(self._on_recognition_done)
        worker.scene_ready.connect(self._on_scene_ready)
        self._start_thread()

    def _launch_placement(self):
        all_reconnus = {**self._session.pending_reconnus}
        print(f"[WORKER] Lancement PlacementWorker | objets: {list(all_reconnus.keys())}")
        worker = PlacementWorker(self._session.original_prompt, all_reconnus)
        if not self._launch_worker(worker):
            return
        worker.scene_ready.connect(self._on_scene_ready)
        self._start_thread()

    def _launch_modify(self, prompt):
        print(f"[WORKER] Lancement ModifySceneWorker | prompt: '{prompt}'")
        worker = ModifySceneWorker(
            prompt,
            self._session.scene_json(),
            self._session.objet_reconnus
        )
        if not self._launch_worker(worker):
            return
        worker.scene_ready.connect(self._on_scene_ready)
        self._start_thread()

    # ------------------------------------------------------------------ #
    #                    CALLBACKS DU PIPELINE                            #
    # ------------------------------------------------------------------ #

    def _cleanup_thread(self):
        print("[THREAD] Thread termine, nettoyage.")
        self._thread = None
        self._worker = None
        if self._pending_intent:
            intent = self._pending_intent
            self._pending_intent = None
            instruction = self._pending_instruction
            if intent == "modify_scene":
                self._launch_modify(instruction)
            else:
                self._launch_recognition(instruction)

    def _on_status_update(self, msg):
        self.status_label.setText(msg)

    def _on_recognition_done(self, objet_reconnus, non_reconnus, alternatives_map):
        print(f"[RECOGNITION] Reconnus: {list(objet_reconnus.keys())} | Non reconnus: {non_reconnus}")
        self._session.pending_reconnus = dict(objet_reconnus)
        self._session.pending_unrecognized = list(non_reconnus)
        self._session.alternatives_map = dict(alternatives_map)

        objects_data = objets_list(OBJETS_DIR)

        if objet_reconnus:
            noms = ", ".join(objet_reconnus.keys())
            self._append_message(f"Objet(s) reconnu(s) : <b>{html.escape(noms)}</b>", "assistant")

        link_color = "#9b3050" if self._current_theme == "pink" else "#7c6af8"
        remove_color = "#c0392b" if self._current_theme == "pink" else "#f87171"

        for label in non_reconnus:
            alts = alternatives_map.get(label, [])
            escaped_label = html.escape(label)
            if alts:
                links = []
                for urdf in alts:
                    name = objects_data[urdf]["name"].strip() if urdf in objects_data else urdf
                    escaped_name = html.escape(name)
                    links.append(
                        f'<a href="resolve:{label}:{urdf}" '
                        f'style="color:{link_color}; text-decoration:underline; cursor:pointer;">'
                        f'{escaped_name}</a>'
                    )
                links.append(
                    f'<a href="remove:{label}" '
                    f'style="color:{remove_color}; text-decoration:underline; cursor:pointer;">'
                    f'Supprimer</a>'
                )
                options = " &nbsp;|&nbsp; ".join(links)
                self._append_message(
                    f'Objet non reconnu : <b>"{escaped_label}"</b><br>'
                    f'Choisissez une alternative : {options}',
                    "assistant"
                )
            else:
                self._append_message(
                    f'Objet non reconnu : <b>"{escaped_label}"</b> — aucune alternative trouvee.'
                    f' <a href="remove:{label}" style="color:{remove_color}; text-decoration:underline;">'
                    f'Supprimer</a>',
                    "assistant"
                )

        self._session.add_message("assistant", f"Objets non reconnus : {', '.join(non_reconnus)}")

        if not objet_reconnus and not non_reconnus:
            self._append_message("Aucun objet reconnu dans la description.", "error")

    def _on_link_clicked(self, url):
        link = url.toString()
        print(f"[LINK] Clic : '{link}'")
        objects_data = objets_list(OBJETS_DIR)

        if link.startswith("resolve:"):
            parts = link.split(":", 2)
            if len(parts) == 3:
                _, label, urdf = parts
                if label in self._session.pending_unrecognized and urdf in objects_data:
                    self._session.resolve_object(label, urdf, objects_data)
                    name = objects_data[urdf]["name"].strip()
                    self._append_message(
                        f'"{html.escape(label)}" remplace par <b>{html.escape(name)}</b>',
                        "system"
                    )

        elif link.startswith("remove:"):
            label = link.split(":", 1)[1]
            if label in self._session.pending_unrecognized:
                self._session.remove_object(label)
                self._append_message(
                    f'"{html.escape(label)}" supprime de la scene.',
                    "system"
                )

        if self._session.all_resolved():
            if self._session.pending_reconnus:
                self._append_message("Tous les objets resolus. Generation de la scene...", "system")
                self._launch_placement()
            else:
                self._append_message("Aucun objet restant. Veuillez decrire une nouvelle scene.", "error")

    def _on_scene_ready(self, objetsList):
        print(f"[SCENE] Scene prete : {[obj.get('id') for obj in objetsList]}")
        self._current_objetsList = objetsList
        self._session.objets_list = objetsList

        objects_data = objets_list(OBJETS_DIR)
        self._session.objet_reconnus = {}
        for obj in objetsList:
            oid = obj.get("id", "")
            urdf = obj.get("urdf", "")
            if urdf in objects_data:
                self._session.objet_reconnus[oid] = {
                    "urdf": urdf,
                    "path": objects_data[urdf]["path"],
                    "dimensions": objects_data[urdf]["dimensions"]
                }

        self.scene_view.update_scene(objetsList)
        names = ", ".join(obj.get("id", "?") for obj in objetsList)
        msg = f"Scene generee avec : {html.escape(names)}."
        self._append_message(msg, "assistant")
        self._session.add_message("assistant", msg)
        self.download_btn.setEnabled(True)
        self.genesis_btn.setEnabled(True)
        self.validate_btn.setEnabled(True)

    def _on_error(self, error_msg):
        self._append_message(html.escape(error_msg), "error")

    def _on_worker_finished(self):
        if self._pending_intent:
            return
        self.send_btn.setEnabled(True)
        self._update_status("Pret", "ready")

    # ------------------------------------------------------------------ #
    #                    NOUVEAU THREAD / ACTIONS                         #
    # ------------------------------------------------------------------ #

    def _on_load_image(self):
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "En cours", "Generation en cours, veuillez patienter.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choisir une ou plusieurs images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not paths:
            return

        self._append_images(paths)

        self.download_btn.setEnabled(False)
        self.genesis_btn.setEnabled(False)
        self.validate_btn.setEnabled(False)
        self._reference_image_path = paths[0]
        worker = ImageSceneWorker(paths)
        if not self._launch_worker(worker):
            return
        worker.scene_ready.connect(self._on_scene_ready)
        self._start_thread()

    def _on_new_thread(self):
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "En cours", "Attendez la fin du traitement en cours.")
            return

        self._session.reset()
        self._current_objetsList = None
        self.chat_display.clear()
        self.scene_view.clear()
        self.scene_view._add_ground()
        self.download_btn.setEnabled(False)
        self.genesis_btn.setEnabled(False)
        self.validate_btn.setEnabled(False)
        self._reference_image_path = None
        self._update_status("Pret", "ready")
        self._append_message("Nouvelle conversation. Decrivez votre scene.", "system")

    def _on_download_scene(self):
        dest, _ = QFileDialog.getSaveFileName(self, "Enregistrer la scene", "scene.json", "JSON (*.json)")
        if dest:
            try:
                shutil.copy2(SCENE_OUTPUT_FILE, dest)
                self._append_message(f"Scene enregistree : {html.escape(dest)}", "system")
            except Exception as e:
                self._append_message(f"Erreur : {html.escape(str(e))}", "error")

    def _on_validate_scene(self):
        if self._current_objetsList is None:
            return
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "En cours", "Generation en cours, veuillez patienter.")
            return

        if self._reference_image_path:
            mode = f"image ({os.path.basename(self._reference_image_path)})"
        else:
            mode = "prompt"
        self._append_message(f"Validation de la scene en cours (mode {html.escape(mode)})...", "system")

        worker = ValidationWorker(
            json_file=SCENE_OUTPUT_FILE,
            prompt=self._session.original_prompt,
            image_path=self._reference_image_path,
        )
        if not self._launch_worker(worker):
            return
        worker.scene_ready.connect(self._on_scene_ready)
        self._start_thread()

    def _on_launch_genesis(self):
        if self._current_objetsList is None:
            return
        launcher = os.path.join(SRC_DIR, "launch_genesis.py")
        try:
            subprocess.Popen(
                [sys.executable, launcher, SCENE_OUTPUT_FILE],
                cwd=SRC_DIR,
                start_new_session=True,
            )
            self._append_message("Simulation Genesis lancee dans une fenetre separee.", "system")
        except Exception as e:
            self._append_message(f"Impossible de lancer Genesis : {html.escape(str(e))}", "error")

    # ------------------------------------------------------------------ #
    #                             THEME                                   #
    # ------------------------------------------------------------------ #

    def _update_status(self, text, state="ready"):
        if self._current_theme == "dark":
            dot_color = "#a6e3a1" if state == "ready" else "#f9e2af"
            text_color = "#585b70"
        else:
            dot_color = "#6aaa82" if state == "ready" else "#d4902a"
            text_color = "#9b6a7a"
        self.status_label.setText(
            f'<font color="{dot_color}">&#9679;</font>'
            f'&nbsp;<font color="{text_color}" style="font-size:11px;">{text}</font>'
        )

    def _toggle_theme(self):
        state = "working" if "cours" in self.status_label.text() else "ready"
        if self._current_theme == "dark":
            self._current_theme = "pink"
            self.theme_btn.setText("🌑")
            self._pink_theme()
        else:
            self._current_theme = "dark"
            self.theme_btn.setText("🌸")
            self._dark_theme()
        self._update_status(
            "En cours..." if state == "working" else "Pret", state
        )

    def _on_text_version_changed(self):
        import pipeline_worker
        version = self.text_version_combo.currentData()
        pipeline_worker.set_text_version(version)

    def _on_image_version_changed(self):
        version = self.image_version_combo.currentData()
        print(f"[VERSION] Pipeline image -> {version}")

    def _dark_theme(self):
        app = QApplication.instance()

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window,         QColor("#11111b"))
        palette.setColor(QPalette.ColorRole.WindowText,     QColor("#cdd6f4"))
        palette.setColor(QPalette.ColorRole.Base,           QColor("#0d0d14"))
        palette.setColor(QPalette.ColorRole.AlternateBase,  QColor("#1e1e2e"))
        palette.setColor(QPalette.ColorRole.Text,           QColor("#cdd6f4"))
        palette.setColor(QPalette.ColorRole.Button,         QColor("#313244"))
        palette.setColor(QPalette.ColorRole.ButtonText,     QColor("#cdd6f4"))
        palette.setColor(QPalette.ColorRole.Highlight,      QColor("#7c3aed"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        app.setPalette(palette)

        app.setStyleSheet("""
            QMainWindow, QWidget  { background: #11111b; }

            QFrame {
                background: #1e1e2e;
                border: 1px solid #2a2a3d;
                border-radius: 12px;
            }

            /* --- Buttons base --- */
            QPushButton {
                background: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 6px 14px;
                font-weight: 500;
            }
            QPushButton:hover   { background: #45475a; border-color: #6272a4; }
            QPushButton:pressed { background: #585b70; }
            QPushButton:disabled { color: #3d3f52; background: #1e1e2e; border-color: #2a2a3d; }

            /* --- Theme toggle --- */
            QPushButton#theme_btn {
                background: #1e1e2e;
                border: 1px solid #45475a;
                border-radius: 17px;
                font-size: 16px;
                padding: 0;
            }
            QPushButton#theme_btn:hover { background: #313244; }

            /* --- Send button (accent) --- */
            QPushButton#send_btn {
                background: #7c3aed;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                font-weight: 600;
            }
            QPushButton#send_btn:hover    { background: #8b5cf6; }
            QPushButton#send_btn:pressed  { background: #6d28d9; }
            QPushButton#send_btn:disabled { background: #2a2a3d; color: #3d3f52; }

            /* --- Image button --- */
            QPushButton#image_btn {
                background: #2d1f47;
                color: #a78bfa;
                border: 1px solid #4c3b7a;
                border-radius: 8px;
                font-weight: 500;
            }
            QPushButton#image_btn:hover    { background: #3b2860; border-color: #7c3aed; }
            QPushButton#image_btn:disabled { background: #1e1e2e; color: #3d3f52; border-color: #2a2a3d; }

            /* --- New thread button --- */
            QPushButton#new_thread_btn {
                background: transparent;
                color: #6272a4;
                border: 1px solid #45475a;
                border-radius: 7px;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton#new_thread_btn:hover { background: #313244; color: #cdd6f4; }

            /* --- Action buttons --- */
            QPushButton#download_btn {
                background: #1a3320;
                color: #4ade80;
                border: 1px solid #16532d;
                font-weight: 600;
                border-radius: 9px;
            }
            QPushButton#download_btn:hover { background: #166534; color: #86efac; }
            QPushButton#download_btn:disabled { background: #1e1e2e; color: #3d3f52; border-color: #2a2a3d; }

            QPushButton#genesis_btn {
                background: #172040;
                color: #60a5fa;
                border: 1px solid #1e3a6e;
                font-weight: 600;
                border-radius: 9px;
            }
            QPushButton#genesis_btn:hover { background: #1e40af; color: #bfdbfe; }
            QPushButton#genesis_btn:disabled { background: #1e1e2e; color: #3d3f52; border-color: #2a2a3d; }

            QPushButton#validate_btn {
                background: #2d1f08;
                color: #fbbf24;
                border: 1px solid #78350f;
                font-weight: 600;
                border-radius: 9px;
            }
            QPushButton#validate_btn:hover { background: #78350f; color: #fde68a; }
            QPushButton#validate_btn:disabled { background: #1e1e2e; color: #3d3f52; border-color: #2a2a3d; }

            /* --- Input --- */
            QLineEdit {
                background: #16161f;
                color: #cdd6f4;
                border: 1.5px solid #45475a;
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 13px;
                selection-background-color: #7c3aed;
            }
            QLineEdit:focus { border-color: #7c3aed; background: #1c1c2e; }

            /* --- Chat display --- */
            QTextBrowser {
                background: #13131d;
                color: #cdd6f4;
                border: 1px solid #2a2a3d;
                border-radius: 10px;
                padding: 6px;
            }

            /* --- Labels --- */
            QLabel              { color: #cdd6f4; background: transparent; border: none; }
            QLabel#panel_title  { color: #a6adc8; font-weight: 700; }
            QLabel#status_label { background: transparent; }

            /* --- Splitter --- */
            QSplitter::handle { background: #2a2a3d; }

            /* --- Scrollbar --- */
            QScrollBar:vertical {
                background: #13131d;
                width: 5px;
                border-radius: 3px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #45475a;
                border-radius: 3px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background: #6272a4; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

            /* --- Version bar --- */
            QFrame#version_bar { background: transparent; border: none; }
            QLabel#version_label {
                color: #45475a;
                font-size: 10px;
                font-weight: 600;
                background: transparent;
            }
            QComboBox#version_combo {
                background: #1e1e2e;
                color: #9399b2;
                border: 1px solid #313244;
                border-radius: 6px;
                padding: 2px 6px 2px 10px;
                font-size: 11px;
                min-height: 22px;
                max-height: 22px;
            }
            QComboBox#version_combo:hover { border-color: #7c3aed; color: #cdd6f4; }
            QComboBox#version_combo::drop-down { border: none; width: 16px; }
            QComboBox#version_combo QAbstractItemView {
                background: #252535;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 7px;
                padding: 3px;
                selection-background-color: #7c3aed;
                selection-color: #ffffff;
                outline: none;
            }
        """)

        self.scene_view.set_background((22, 22, 35, 255))
        self._update_status("Pret", "ready")

    def _pink_theme(self):
        app = QApplication.instance()

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window,          QColor("#fdf5f7"))
        palette.setColor(QPalette.ColorRole.WindowText,      QColor("#3d1a26"))
        palette.setColor(QPalette.ColorRole.Base,            QColor("#fff9fa"))
        palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#fdf5f7"))
        palette.setColor(QPalette.ColorRole.Text,            QColor("#3d1a26"))
        palette.setColor(QPalette.ColorRole.Button,          QColor("#fce4ea"))
        palette.setColor(QPalette.ColorRole.ButtonText,      QColor("#9b3050"))
        palette.setColor(QPalette.ColorRole.Highlight,       QColor("#c06070"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        app.setPalette(palette)

        app.setStyleSheet("""
            QMainWindow, QWidget { background: #fdf5f7; }

            QFrame {
                background: #fff9fa;
                border: 1px solid #eecdd8;
                border-radius: 12px;
            }

            /* --- Buttons base --- */
            QPushButton {
                background: #fce4ea;
                color: #9b3050;
                border: 1px solid #f5c0ce;
                border-radius: 8px;
                padding: 6px 14px;
                font-weight: 500;
            }
            QPushButton:hover   { background: #f9d4dd; border-color: #e898b0; }
            QPushButton:pressed { background: #f5c4d0; }
            QPushButton:disabled { color: #dba8b8; background: #fdf5f7; border-color: #f0dae0; }

            /* --- Theme toggle --- */
            QPushButton#theme_btn {
                background: #fce4ea;
                border: 1px solid #f5c0ce;
                border-radius: 17px;
                font-size: 16px;
                padding: 0;
            }
            QPushButton#theme_btn:hover { background: #f9d4dd; }

            /* --- Send button (accent) --- */
            QPushButton#send_btn {
                background: #c06070;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                font-weight: 600;
            }
            QPushButton#send_btn:hover    { background: #a84f5e; }
            QPushButton#send_btn:pressed  { background: #90404e; }
            QPushButton#send_btn:disabled { background: #f0dae0; color: #dba8b8; }

            /* --- Image button --- */
            QPushButton#image_btn {
                background: #f4e8f8;
                color: #7a3a8a;
                border: 1px solid #e0c4ec;
                border-radius: 8px;
                font-weight: 500;
            }
            QPushButton#image_btn:hover    { background: #ecd8f5; border-color: #c090dc; }
            QPushButton#image_btn:disabled { background: #fdf5f7; color: #dba8b8; border-color: #f0dae0; }

            /* --- New thread button --- */
            QPushButton#new_thread_btn {
                background: transparent;
                color: #c4909e;
                border: 1px solid #f0c8d4;
                border-radius: 7px;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton#new_thread_btn:hover { background: #fce4ea; color: #9b3050; }

            /* --- Action buttons --- */
            QPushButton#download_btn {
                background: #eef8f2;
                color: #2d6a45;
                border: 1px solid #b8dcc8;
                font-weight: 600;
                border-radius: 9px;
            }
            QPushButton#download_btn:hover { background: #ddf0e6; color: #1e5234; }
            QPushButton#download_btn:disabled { background: #fdf5f7; color: #dba8b8; border-color: #f0dae0; }

            QPushButton#genesis_btn {
                background: #eef2f9;
                color: #2a3a7a;
                border: 1px solid #c0ccec;
                font-weight: 600;
                border-radius: 9px;
            }
            QPushButton#genesis_btn:hover { background: #dde5f7; color: #1e2d68; }
            QPushButton#genesis_btn:disabled { background: #fdf5f7; color: #dba8b8; border-color: #f0dae0; }

            QPushButton#validate_btn {
                background: #f5eef9;
                color: #6a3080;
                border: 1px solid #dcc0ec;
                font-weight: 600;
                border-radius: 9px;
            }
            QPushButton#validate_btn:hover { background: #ecddf7; color: #542465; }
            QPushButton#validate_btn:disabled { background: #fdf5f7; color: #dba8b8; border-color: #f0dae0; }

            /* --- Input --- */
            QLineEdit {
                background: #fff4f6;
                color: #3d1a26;
                border: 1.5px solid #f0c8d4;
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 13px;
                selection-background-color: #f5c0ce;
            }
            QLineEdit:focus { border-color: #c06070; background: #fff9fa; }

            /* --- Chat display --- */
            QTextBrowser {
                background: #fff9fa;
                color: #3d1a26;
                border: 1px solid #eecdd8;
                border-radius: 10px;
                padding: 6px;
            }

            /* --- Labels --- */
            QLabel              { color: #3d1a26; background: transparent; border: none; }
            QLabel#panel_title  { color: #7a2240; font-weight: 700; }
            QLabel#status_label { background: transparent; }

            /* --- Splitter --- */
            QSplitter::handle { background: #eecdd8; width: 1px; }

            /* --- Scrollbar --- */
            QScrollBar:vertical {
                background: #fdf5f7;
                width: 5px;
                border-radius: 3px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #f0c8d4;
                border-radius: 3px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background: #c06070; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

            /* --- Version bar --- */
            QFrame#version_bar { background: transparent; border: none; }
            QLabel#version_label {
                color: #c4909e;
                font-size: 10px;
                font-weight: 600;
                background: transparent;
            }
            QComboBox#version_combo {
                background: #fce4ea;
                color: #9b3050;
                border: 1px solid #f0c8d4;
                border-radius: 6px;
                padding: 2px 6px 2px 10px;
                font-size: 11px;
                min-height: 22px;
                max-height: 22px;
            }
            QComboBox#version_combo:hover { border-color: #c06070; color: #7a2240; }
            QComboBox#version_combo::drop-down { border: none; width: 16px; }
            QComboBox#version_combo QAbstractItemView {
                background: #fff9fa;
                color: #3d1a26;
                border: 1px solid #f0c8d4;
                border-radius: 7px;
                padding: 3px;
                selection-background-color: #c06070;
                selection-color: #ffffff;
                outline: none;
            }
        """)

        self.scene_view.set_background((245, 235, 240, 255))
        self._update_status("Pret", "ready")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PAI2D Real-to-Sim")
    font = app.font()
    font.setPointSize(13)
    app.setFont(font)
    window = MainWindow()
    print("[APP] PAI2D Real-to-Sim lance")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
