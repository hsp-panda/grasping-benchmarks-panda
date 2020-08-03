// QTIFW Script for non-interactively installing QtCreator
//
// Author: Diego Ferigo <diego.ferigo@iit.it>
//
// Sources:
//
// - https://doc.qt.io/qtinstallerframework/noninteractive.html
// - https://github.com/benlau/qtci/blob/master/bin/extract-qt-installer
// - https://stackoverflow.com/questions/25105269/silent-install-qt-run-installer-on-ubuntu-server

function Controller() {
}

Controller.prototype.WelcomePageCallback = function() {
    gui.clickButton(buttons.NextButton, 2000);
}

Controller.prototype.CredentialsPageCallback = function() {
    gui.clickButton(buttons.NextButton);
}

Controller.prototype.IntroductionPageCallback = function() {
    gui.clickButton(buttons.NextButton);
}

Controller.prototype.ObligationsPageCallback = function()
{
    var page = gui.pageWidgetByObjectName("ObligationsPage");
    page.obligationsAgreement.setChecked(true);
    page.completeChanged();
    gui.clickButton(buttons.NextButton);
}

Controller.prototype.TargetDirectoryPageCallback = function() {
   gui.currentPageWidget().TargetDirectoryLineEdit.setText("/opt/qtcreator");
   gui.clickButton(buttons.NextButton);
}

Controller.prototype.ComponentSelectionPageCallback = function() {
    gui.clickButton(buttons.NextButton);
}

Controller.prototype.LicenseAgreementPageCallback = function() {
    gui.currentPageWidget().AcceptLicenseRadioButton.setChecked(true);
    gui.clickButton(buttons.NextButton);
}

Controller.prototype.ReadyForInstallationPageCallback = function() {
    gui.clickButton(buttons.CommitButton);
}

Controller.prototype.FinishedPageCallback = function() {
    // Minimal platform doesn't have this checkbox
    var widget = gui.currentPageWidget();
    if (widget.LaunchQtCreatorCheckBoxForm) {
        widget.LaunchQtCreatorCheckBoxForm.launchQtCreatorCheckBox.setChecked(false);
    }
    gui.clickButton(buttons.FinishButton);
}
